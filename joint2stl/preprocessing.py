"""Conservative patient outline for excluding table/supports from bone masks.

Work from the complete acquired slice, even when the bone ROI is small. The
source HU array is never changed. Small 2-D working images bound memory use.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .model import Box, Volume


def patient_region(volume: Volume, roi: Box, progress=lambda *_: None):
    if volume.is_mr:raise ValueError('CT 人体密度轮廓清理不适用于 MRI。')
    spacing = volume.spacing_zyx[1:]
    stride = np.maximum(1, np.ceil(1.2 / spacing).astype(int))
    coarse_spacing = spacing * stride
    shape = tuple(s.stop - s.start for s in roi)
    keep = np.ones(shape, dtype=bool)
    unsupported = []
    removed_dense = 0
    supported = 0
    core_radius = 3.5
    for index, z in enumerate(range(roi[0].start, roi[0].stop)):
        if index % 8 == 0:
            progress(int(index / shape[0] * 95), f'识别人体轮廓并排除床板 {index + 1} / {shape[0]} 层')
        full = volume.hu[z]
        image = full[::stride[0], ::stride[1]]
        # Pad explicitly: a cropped body at the image boundary must not gain an
        # artificial, unbounded thickness from distance_transform_edt.
        # Dense plaster, carbon plates and metal braces must not become part of
        # the body outline merely because they touch skin. Recover enclosed
        # cortex/marrow by filling holes in the soft-tissue region instead.
        soft = (image > -350) & (image < 150)
        radius = np.ceil(4 / coarse_spacing).astype(int)
        gy, gx = np.ogrid[-radius[0]:radius[0]+1, -radius[1]:radius[1]+1]
        disk = (gy*coarse_spacing[0])**2 + (gx*coarse_spacing[1])**2 <= 4**2
        padded = np.pad(soft, [(int(r), int(r)) for r in radius], mode='edge')
        closed = ndi.binary_closing(padded, structure=disk)
        closed = closed[radius[0]:radius[0]+soft.shape[0], radius[1]:radius[1]+soft.shape[1]]
        # Close narrow high-density channels from pins before hole filling;
        # otherwise a cortex connected to hardware can be mistaken for air.
        tissue = ndi.binary_fill_holes(closed | soft)
        distance = ndi.distance_transform_edt(np.pad(tissue, 1), sampling=coarse_spacing)[1:-1, 1:-1]
        cores, _ = ndi.label(distance >= core_radius, structure=np.ones((3, 3), bool))
        boxes = ndi.find_objects(cores)
        selected = np.zeros(image.shape, bool)
        for ident, box in enumerate(boxes, 1):
            if box is None:
                continue
            yy, xx = np.nonzero(cores[box] == ident)
            if len(yy) * np.prod(coarse_spacing) < 120:
                continue
            points = np.column_stack((yy * coarse_spacing[0], xx * coarse_spacing[1]))
            eigenvalues = np.linalg.eigvalsh(np.cov(points.T))
            widths = np.sqrt(np.maximum(eigenvalues, 0))
            values = image[box][yy, xx]
            # Broad, thin sheets and rails do not qualify as a patient core.
            # Retain multiple cores so both legs/arms survive this step.
            if widths[0] < 2.5 or widths[-1] / max(widths[0], .1) > 10:
                continue
            if np.mean((values > -350) & (values < 250)) < .12:
                continue
            selected[box] |= cores[box] == ident
        if not selected.any():
            # Fail open on uncertain slices. Never erase a whole slice just
            # because no sufficiently thick soft-tissue core was found.
            unsupported.append(z + 1)
            continue
        supported += 1
        envelope = tissue & (ndi.distance_transform_edt(~selected, sampling=coarse_spacing) <= core_radius + 4)
        envelope = ndi.binary_fill_holes(envelope)
        envelope = ndi.distance_transform_edt(~envelope, sampling=coarse_spacing) <= 1.5
        full_keep = np.repeat(np.repeat(envelope, stride[0], axis=0), stride[1], axis=1)[:full.shape[0], :full.shape[1]]
        local = full_keep[roi[1:]]
        keep[index] = local
        removed_dense += int(np.count_nonzero((full[roi[1:]] >= 200) & ~local))
    if not supported:
        raise ValueError('没有可靠的人体轮廓，已保留当前分割。请缩小至关节范围后重试。')
    progress(100, '人体轮廓检查完成')
    report = {
        'method': 'soft_tissue_core_v2', 'scope': 'bone segmentation only; source HU preserved',
        'roi_zyx': [[s.start, s.stop] for s in roi],
        'supported_slices': supported, 'preserved_uncertain_slices_1based': unsupported,
        'excluded_dense_voxels': removed_dense,
    }
    return keep, report


def exclude_external_attachments(mask, patient_mask, spacing_zyx, progress=lambda *_: None):
    """Drop exterior components; protect anything anchored inside the patient.

    A skin-adjacent plaster shell may partially lie in the protective outline
    margin. Remove its complete component only when it has no deep-body anchor.
    External fixators connected to bone remain flagged instead of being treated
    as a confidently separate cast or brace.
    """
    if mask.shape != patient_mask.shape:
        raise ValueError('外部结构检查范围不一致。')
    components, _ = ndi.label(mask, structure=ndi.generate_binary_structure(3, 3))
    counts = np.zeros(int(components.max()) + 1, np.int64)
    inside = np.zeros_like(counts); anchored = np.zeros_like(counts)
    for z in range(mask.shape[0]):
        if z % 16 == 0:
            progress(int(z / mask.shape[0] * 95), f'检查石膏、夹板和支具 {z + 1} / {mask.shape[0]} 层')
        plane = components[z]
        values = np.bincount(plane.ravel(), minlength=len(counts)); counts += values
        values = np.bincount(plane[patient_mask[z]], minlength=len(counts)); inside += values
        depth = ndi.distance_transform_edt(np.pad(patient_mask[z], 1), sampling=spacing_zyx[1:])[1:-1, 1:-1]
        values = np.bincount(plane[depth > 5], minlength=len(counts)); anchored += values
    exterior = (counts - inside) / np.maximum(counts, 1)
    candidates = (exterior > .35) & (counts > 0); candidates[0] = False
    removable = candidates & (anchored == 0)
    ambiguous = candidates & (anchored > 0)
    excluded = removable[components]
    filtered = mask & patient_mask & ~excluded
    report = {'attachment_method': 'exterior_components_with_deep_body_anchor_v1',
              'removed_external_components': int(np.count_nonzero(removable)),
              'ambiguous_attachment_components': int(np.count_nonzero(ambiguous)),
              'removed_external_component_voxels': int(np.count_nonzero(mask & excluded)),
              'removed_labeled_voxels': int(np.count_nonzero(mask & ~filtered)),
              'external_attachment_scope': 'plaster/splints/braces outside soft-tissue outline; bone-connected devices require review'}
    progress(100, '外部附着物检查完成')
    return filtered, report


def remove_segmented_table(session, progress=lambda *_: None):
    if not session.segments or not np.any(session.labels[session.roi]):
        raise ValueError('请先运行自动初分割；初分割默认启用自动去床板。')
    keep, report = patient_region(session.volume, session.roi, lambda value,text:progress(int(value*.55),text))
    local = session.labels[session.roi].copy()
    cleaned, attachments = exclude_external_attachments(local > 0, keep, session.volume.spacing_zyx,
        lambda value,text:progress(55+int(value*.45),text))
    removed = (local > 0) & ~cleaned
    report.update(attachments)
    report['removed_labeled_voxels'] = int(np.count_nonzero(removed))
    report['affected_labels'] = [int(x) for x in np.unique(local[removed])]
    local[removed] = 0
    return local, report
