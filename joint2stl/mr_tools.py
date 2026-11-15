"""Unreviewed MRI annotation proposals from the generic promptable LiteMedSAM.

Candidates are uint16 arrays in ``session.roi`` coordinates, suitable for a later
``Session.apply``. Prediction/refinement never commits masks, presence, review
state, reports, or history. Only ``ensure_mr_segments`` edits the session.

Plane names retain the UI's array-axis convention, not an anatomical claim for
oblique acquisitions. Box coordinates are half-open XYXY in the *full raw*
``np.take(volume.values, index, axis=fixed_axis)`` image. Drag endpoints map
directly to their min/max coordinates; display flips must already be inverted.
"""
from __future__ import annotations

from itertools import product
from numbers import Integral

import numpy as np
from scipy import ndimage as ndi

from . import ai
from .model import PALETTE, Segment
from .mr_policy import require_mri_modeling_source,modeling_source_metadata


MR_STRUCTURES = [
    ('femur', '股骨', 'bone'),
    ('tibia', '胫骨', 'bone'),
    ('fibula', '腓骨', 'bone'),
    ('patella', '髌骨', 'bone'),
    ('acl', '前交叉韧带 ACL', 'ligament'),
    ('pcl', '后交叉韧带 PCL', 'ligament'),
]

_PLANES = {'axial': 0, 'coronal': 1, 'sagittal': 2}
_PERCENTILES = (1., 99.)
_PROMPT_MARGIN_MM = 2.
_LIMITATIONS = (
    'Prompt-assisted annotation, not automatic anatomical identification or a '
    'diagnosis. Model scores are UNCALIBRATED; consistency with a prior is not '
    'ground-truth accuracy. Presence is user supplied and is never inferred. '
    'All candidates require review.'
)


def _require_mr(session):
    if session.volume.modality != 'MR':
        raise ValueError('MRI annotation requires an MR volume.')


def _segment(session, ident):
    if not isinstance(ident, Integral) or isinstance(ident, (bool, np.bool_)):
        raise ValueError('Select an existing MRI label.')
    segment = session.segment(int(ident))
    if segment is None or not 1 <= ident <= 65535:
        raise ValueError('Select an existing MRI label.')
    if segment.kind not in ('bone', 'ligament'):
        raise ValueError('MRI annotation supports bone and ligament labels.')
    return segment


def ensure_mr_segments(session, *, acl_absent=False):
    """Add missing structure labels; only literal ``True`` declares ACL absence.

    Existing structure keys (or exact, unassigned catalogue names) are reused.
    New labels contain no voxels and default to unknown presence. An explicit
    absence uses Session.update_presence, including its atomic mask/metadata
    undo. False never resets a previously recorded presence decision.
    """
    _require_mr(session)
    if type(acl_absent) is not bool:
        raise ValueError('acl_absent must be an explicit bool.')
    found = {}
    for key, name, _ in MR_STRUCTURES:
        found[key] = next((s for s in session.segments if s.structure == key), None)
        if found[key] is None:
            found[key] = next((s for s in session.segments
                               if not s.structure and s.name == name), None)
    # Do not allocate an ID already present as an orphan voxel label.
    next_id = max(session.next_segment_id(), int(session.labels.max()) + 1)
    if next_id + sum(s is None for s in found.values()) - 1 > 65535:
        raise ValueError('MRI label IDs exceed uint16 capacity.')
    for key, name, kind in MR_STRUCTURES:
        segment = found[key]
        if segment is None:
            segment = Segment(next_id, name, PALETTE[(next_id - 1) % len(PALETTE)],
                              kind=kind, structure=key)
            session.segments.append(segment)
            found[key] = segment
            next_id += 1
            session.dirty = True
        elif (segment.structure, segment.kind) != (key, kind):
            segment.structure, segment.kind = key, kind
            session.dirty = True
    result = {key: segment.id for key, segment in found.items()}
    if acl_absent is True:
        session.update_presence(result['acl'], 'absent', source='user_stated')
    return result


def _view(array, axis, index):
    """A writable array plane; unlike np.take this does not copy labels."""
    selection = [slice(None)] * 3
    selection[axis] = index
    return array[tuple(selection)]


def _point(point, shape):
    try:
        values = tuple(point)
    except TypeError as exc:
        raise ValueError('Prompt endpoints must be voxel ZYX integer triples.') from exc
    if len(values) != 3 or any(not isinstance(v, Integral) or isinstance(v, (bool, np.bool_))
                               for v in values):
        raise ValueError('Prompt endpoints must be voxel ZYX integer triples.')
    if any(not 0 <= v < n for v, n in zip(values, shape)):
        raise ValueError('Prompt endpoints must be inside the volume.')
    return tuple(int(v) for v in values)


def _local_box(box, roi, axes):
    y_axis, x_axis = axes
    x0, y0, x1, y1 = box
    return [x0 - roi[x_axis].start, y0 - roi[y_axis].start,
            x1 - roi[x_axis].start, y1 - roi[y_axis].start]


def _scope(shape, box, spacing, margin_mm):
    """Pixel centres within a physical Euclidean distance of the prompt box."""
    x0, y0, x1, y1 = box
    y, x = np.ogrid[:shape[0], :shape[1]]
    dy = np.maximum(np.maximum(y0 - y, y - (y1 - 1)), 0) * spacing[0]
    dx = np.maximum(np.maximum(x0 - x, x - (x1 - 1)), 0) * spacing[1]
    return dy * dy + dx * dx <= margin_mm * margin_mm + 1e-9


def _geometry(volume, roi, plane, index, box):
    axis = _PLANES[plane]
    axes = [a for a in range(3) if a != axis]
    x0, y0, x1, y1 = box
    lo, hi = np.zeros(3), np.zeros(3)
    lo[axis], hi[axis] = index, index + 1
    lo[axes], hi[axes] = [y0, x0], [y1, x1]
    corners = np.array(list(product(*zip(lo - .5, hi - .5))))
    world = volume.world(corners)
    direction=np.zeros(3);direction[axis]=1
    normal=volume.world(direction)-volume.world(np.zeros(3))
    normalized=np.abs(normal)/np.linalg.norm(normal)
    return {
        'plane': plane, 'array_plane':plane,'sample_plane': plane, 'fixed_axis': axis,
        'anatomical_plane_estimate':('sagittal','coronal','axial')[int(normalized.argmax())],
        'oblique_plane':bool(normalized.max()<.999),
        'in_plane_axes_zyx': axes, 'native_acquisition_plane': axis == 0,
        'fixed_index_0based': int(index), 'slice_1based': int(index + 1),
        'box_xyxy_0based': [int(v) for v in box],
        'box_zyx_0based': [[int(a), int(b)] for a, b in zip(lo, hi)],
        'image_shape_yx': [int(volume.values.shape[a]) for a in axes],
        'roi_zyx_0based': [[r.start, r.stop] for r in roi],
        'bounds_lps_mm': [world.min(axis=0).tolist(), world.max(axis=0).tolist()],
        'box_corners_lps_mm': world.tolist(),
        'lps_bounds_convention': 'outer voxel faces; all eight affine-transformed corners',
    }


def _report(session, segment, plane, method, device):
    return {
        'method': method, 'model': 'LiteMedSAM', 'modality': 'MR',
        'checkpoint_sha256': ai.CHECKPOINT_SHA256,
        'label_id': int(segment.id), 'label_name': segment.name,
        'kind': segment.kind, 'structure': segment.structure,
        'presence': segment.presence, 'presence_source': segment.presence_source,
        'reviewed': False, 'device': 'not_run', 'requested_device': device,
        'plane': plane, 'sample_plane': plane, 'fixed_axis': _PLANES[plane],
        'native_acquisition_plane': _PLANES[plane] == 0,
        'fixed_indices_0based': [], 'slices': [],
        'accepted_slices': 0, 'flagged_slices': 0, 'warning_slices': 0,
        'model_score_calibration': 'UNCALIBRATED', 'intensity_units': 'MR signal',
        'box_convention': 'full raw image XYXY, half-open; drag endpoints use min/max',
        'roi_zyx_0based': [[r.start, r.stop] for r in session.roi],
        'limitations': _LIMITATIONS,
        'modeling_source':modeling_source_metadata(session),
    }


def _infer(session, axis, index, box, device):
    # np.take preserves the acquired array order in every plane, with no UI flip.
    raw = np.take(session.volume.values, index, axis=axis)
    finite = np.isfinite(raw)
    if not finite.any():
        raise ValueError('MRI slice has no finite signal values.')
    low, high = (float(v) for v in np.percentile(raw[finite], _PERCENTILES))
    degenerate = high <= low
    if degenerate:
        half_width = max(1., abs(low) * 1e-6)
        low, high = low - half_width, low + half_width
    if not np.isfinite([low, high]).all():
        raise ValueError('MRI signal window is not finite.')
    # Sanitize a private image; neither model code nor normalization can edit input.
    image = np.clip(np.nan_to_num(raw, nan=low, posinf=high, neginf=low), low, high)
    image = image.astype(np.float32)
    if not np.isfinite(image).all():
        raise ValueError('MRI signal is outside the inference numeric range.')
    probability, score, actual_device = ai.infer_slice(image, list(box), device, window=(low, high))
    probability = np.asarray(probability)
    if probability.shape != image.shape:
        raise ValueError('MRI model returned an incorrect image shape.')
    score = float(score)
    normalization = {
        'method': 'per_image_finite_percentile_window',
        'percentiles': list(_PERCENTILES), 'window': [low, high],
        'degenerate_percentiles': bool(degenerate),
        'nonfinite_pixels_replaced': int(raw.size - finite.sum()),
    }
    return probability, score, actual_device, normalization


def _screen(probability, prior, available, scope, box, spacing, *, growth_mm=None):
    flags = []
    quality = {'accepted': False, 'flags': flags}
    if not np.isfinite(probability).all():
        flags.append('nonfinite_model_output')
        return prior.copy(), quality
    raw = (probability >= .5) & available & scope
    components, count = ndi.label(raw)
    if prior.any():
        overlaps = np.bincount(components[prior], minlength=count + 1)
        overlaps[0] = 0
        kept = np.flatnonzero(overlaps)
    else:
        # A new annotation must touch the central half of the user prompt.
        x0, y0, x1, y1 = box
        cy0, cx0 = y0 + (y1 - y0) // 4, x0 + (x1 - x0) // 4
        cy1, cx1 = y1 - (y1 - y0) // 4, x1 - (x1 - x0) // 4
        overlaps = np.bincount(components[cy0:cy1, cx0:cx1].ravel(), minlength=count + 1)
        overlaps[0] = 0
        kept = [int(overlaps.argmax())] if overlaps.any() else []
    selected = np.isin(components, kept) if len(kept) else np.zeros_like(raw)
    if not selected.any():
        flags.append('no_component_overlaps_prior_or_prompt_centre')
        return prior.copy(), quality
    x0, y0, x1, y1 = box
    ys, xs = np.where(selected)
    if ys.min() <= y0 or ys.max() >= y1 - 1 or xs.min() <= x0 or xs.max() >= x1 - 1:
        flags.append('touches_bbox')
    quality['area_fraction_of_prompt_box'] = float(selected.sum() / ((x1 - x0) * (y1 - y0)))
    overlarge = quality['area_fraction_of_prompt_box'] >= .85
    if overlarge:
        flags.append('overlarge_prediction')
    prediction = selected.copy()
    if growth_mm is not None:
        prediction &= ndi.distance_transform_edt(~prior, sampling=spacing) <= growth_mm + 1e-9
        prior_area = int(prior.sum())
        raw_ratio = float(selected.sum() / prior_area)
        ratio = float(prediction.sum() / prior_area)
        overlap = int(np.count_nonzero(prediction & prior))
        dice = float(2 * overlap / max(1, int(prediction.sum()) + prior_area))
        coverage = float(overlap / prior_area)
        quality.update(raw_model_area_ratio=raw_ratio, area_ratio_to_initial=ratio,
                       dice_against_initial_not_ground_truth=dice,
                       prior_overlap_fraction=coverage, max_model_growth_mm=growth_mm)
        if not (.5 <= raw_ratio <= 2 and .5 <= ratio <= 2):
            flags.append('area_ratio_outside_0.5_to_2')
        if dice < .5 or coverage < .5:
            flags.append('low_overlap_with_prior')
        quality['accepted'] = not any(f in flags for f in
                                     ('area_ratio_outside_0.5_to_2', 'low_overlap_with_prior'))
    else:
        # A nearly filled box without a prior is not evidence for a new structure.
        quality['accepted'] = not (overlarge and not prior.any())
    return prediction, quality


def _sample(session, candidate, owners, segment, plane, index, box, device, growth_mm=None):
    axis = _PLANES[plane]
    axes = [a for a in range(3) if a != axis]
    roi = session.roi
    current = _view(candidate, axis, index - roi[axis].start)
    original = _view(owners, axis, index - roi[axis].start)
    spacing = np.asarray(session.volume.spacing_zyx)[axes]
    local_box = _local_box(box, roi, axes)
    margin = _PROMPT_MARGIN_MM if growth_mm is None else growth_mm
    scope = _scope(current.shape, local_box, spacing, margin)
    available = ((current == 0) | (current == segment.id)) & ((original == 0) | (original == segment.id))
    prior = (current == segment.id) & scope
    probability, score, actual_device, normalization = _infer(session, axis, index, box, device)
    probability = probability[tuple(roi[a] for a in axes)]
    prediction, quality = _screen(probability, prior, available, scope, local_box, spacing,
                                  growth_mm=growth_mm)
    # A generic prompt model can select posterior soft tissue instead of the
    # low-signal PCL band. Screen new sagittal proposals conservatively. Relative
    # signal contrasts stay invariant to an additive offset/positive scaling;
    # these checks do not determine presence or injury and can reject abnormal
    # or poorly visible ligaments, which remain available for manual annotation.
    step=np.zeros(3);step[axis]=1
    normal=session.volume.world(step)-session.volume.world(np.zeros(3))
    sagittal=abs(normal[0])/np.linalg.norm(normal)>.9
    if segment.structure=='pcl' and sagittal and growth_mm is None and quality['accepted']:
        coords=np.argwhere(prediction)*spacing
        eigen=np.linalg.eigvalsh(np.cov(coords.T)) if len(coords)>2 else np.array([1.,1.])
        elongation=float(np.sqrt(max(0,float(eigen[-1]))/max(1e-9,float(eigen[0]))))
        image=_view(session.volume.values,axis,index)[tuple(roi[a] for a in axes)]
        x0,y0,x1,y1=local_box;window=image[y0:y1,x0:x1]
        low,median,high=(float(v) for v in np.percentile(window,[10,50,90]))
        selected_median=float(np.median(image[prediction]))
        contrast_supported=high>low and selected_median<=median-.1*(high-low)
        quality.update(pcl_sagittal_axis_ratio=elongation,pcl_prompt_signal_percentiles=[low,median,high],
                       pcl_selected_signal_median=selected_median,pcl_relative_signal_supported=bool(contrast_supported))
        if elongation<2.5 or not contrast_supported:
            quality['accepted']=False;quality['flags'].append('pcl_signal_or_shape_uncertain')
    if not np.isfinite(score):
        quality['flags'].append('nonfinite_model_score')
    entry = {
        **_geometry(session.volume, roi, plane, index, box), **quality,
        'model_score_uncalibrated': score if np.isfinite(score) else None,
        'model_score_calibration': 'UNCALIBRATED', 'reviewed': False,
        'device': actual_device, 'normalization': normalization,
    }
    if quality['accepted']:
        current[(current == segment.id) & scope] = 0
        current[prediction & available] = segment.id
    return entry


def _finish(report, before, after):
    entries = report['slices']
    report['fixed_indices_0based'] = [e['fixed_index_0based'] for e in entries]
    report['accepted_slices'] = sum(e['accepted'] for e in entries)
    # Existing UI calls flagged slices "not adopted". Warnings on an accepted
    # proposal remain in its entry and a separate count, without double counting.
    report['flagged_slices'] = sum(not e['accepted'] for e in entries)
    report['warning_slices'] = sum(bool(e['flags']) for e in entries)
    report['inference_performed'] = bool(entries)
    report['device'] = entries[0]['device'] if entries else 'not_run'
    ident = report['label_id']
    report['added_voxels'] = int(np.count_nonzero((after == ident) & (before != ident)))
    report['removed_voxels'] = int(np.count_nonzero((before == ident) & (after != ident)))
    report.setdefault('status', 'candidate' if report['accepted_slices'] else 'unchanged')
    return report


def predict_mr_prompt(session, ident, plane, start_zyx, end_zyx,
                      progress=lambda *_: None, device='auto', span=0):
    """Propose a prompt mask on one raw plane, or an explicit ±span (0..4).

    Absence rejects before any model call. Endpoints must share the fixed index;
    the box is clipped to the ROI and needs at least two pixels in each axis.
    Edits affect only this target in the prompt box plus a physical 2 mm halo.
    Empty/unrelated outputs preserve the prior. Exceptions leave input untouched.
    Reports and entries carry plane/fixed_axis for the UI's slice navigation.
    """
    _require_mr(session)
    segment = _segment(session, ident)
    if segment.presence == 'absent':
        raise ValueError('This label is marked absent; change presence before MRI annotation.')
    if plane not in _PLANES:
        raise ValueError('MRI plane must be axial, coronal, or sagittal.')
    require_mri_modeling_source(session,plane)
    if not isinstance(span, Integral) or isinstance(span, (bool, np.bool_)) or not 0 <= span <= 4:
        raise ValueError('MRI span must be an explicit integer from 0 to 4.')
    axis = _PLANES[plane]
    axes = [a for a in range(3) if a != axis]
    start = _point(start_zyx, session.volume.values.shape)
    end = _point(end_zyx, session.volume.values.shape)
    if start[axis] != end[axis]:
        raise ValueError('Prompt endpoints must lie on the same fixed slice.')
    roi = session.roi
    index = start[axis]
    if not roi[axis].start <= index < roi[axis].stop:
        raise ValueError('The prompt slice is outside the ROI.')
    y_axis, x_axis = axes
    box = [max(roi[x_axis].start, min(start[x_axis], end[x_axis])),
           max(roi[y_axis].start, min(start[y_axis], end[y_axis])),
           min(roi[x_axis].stop, max(start[x_axis], end[x_axis])),
           min(roi[y_axis].stop, max(start[y_axis], end[y_axis]))]
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        raise ValueError('The effective ROI prompt box needs at least 2 pixels per axis.')
    indices = range(max(roi[axis].start, index - span), min(roi[axis].stop, index + span + 1))
    owners = session.labels[roi].copy()
    candidate = owners.copy()
    report = _report(session, segment, plane, 'mr_prompt_assisted_annotation_v1', device)
    report.update(span=int(span), prompt_start_zyx=list(start), prompt_end_zyx=list(end),
                  box_xyxy_0based=box, prompt_margin_mm=_PROMPT_MARGIN_MM)
    for number, fixed in enumerate(indices):
        progress(int(100 * number / len(indices)), f'MRI prompt {number + 1}/{len(indices)}')
        report['slices'].append(_sample(session, candidate, owners, segment, plane, fixed, box, device))
    _finish(report, owners, candidate)
    progress(100, 'Unreviewed MRI annotation candidate ready')
    return candidate, report


def refine_mr_many(session, ids, progress=lambda *_: None, device='auto'):
    """Refine existing masks in native acquisition planes (array axis 0).

    Model boxes come from each prior slice. Growth is at most 1 mm for ligaments
    and 2 mm for bones, with area/overlap screens and no automatic filling. Absent
    labels never invoke the model; stale absent voxels are cleared only in the
    returned ROI candidate. Their original ownership stays reserved throughout
    the batch, including after removal. Empty labels are reported and skipped.
    """
    _require_mr(session)
    require_mri_modeling_source(session)
    requested = list(ids)
    if not requested:
        raise ValueError('Select MRI labels to refine.')
    segments = [_segment(session, ident) for ident in requested]
    segments = list({s.id: s for s in segments}.values())
    roi = session.roi
    owners = session.labels[roi].copy()
    candidate = owners.copy()
    reports = []
    for number, segment in enumerate(segments):
        report = _report(session, segment, 'axial', 'mr_prior_assisted_refinement_v1', device)
        if segment.presence == 'absent':
            candidate[candidate == segment.id] = 0
            report['status'] = 'skipped_absent'
        else:
            growth = 1. if segment.kind == 'ligament' else 2.
            report['max_model_growth_mm'] = growth
            positions = np.flatnonzero((candidate == segment.id).any(axis=(1, 2)))
            if not len(positions):
                report['status'] = 'skipped_empty'
            for slice_number, local_z in enumerate(positions):
                progress(int(100 * (number + slice_number / len(positions)) / len(segments)),
                         f'MRI refinement {number + 1}/{len(segments)} · slice {slice_number + 1}/{len(positions)}')
                prior = candidate[local_z] == segment.id
                ys, xs = np.where(prior)
                # Padding supplies model context; the final Euclidean prior-distance
                # cap, rather than rounded pixel padding, controls allowable growth.
                pad_y, pad_x = np.maximum(1, np.ceil(growth / session.volume.spacing_zyx[1:]).astype(int))
                box = [int(max(0, xs.min() - pad_x) + roi[2].start),
                       int(max(0, ys.min() - pad_y) + roi[1].start),
                       int(min(prior.shape[1], xs.max() + pad_x + 1) + roi[2].start),
                       int(min(prior.shape[0], ys.max() + pad_y + 1) + roi[1].start)]
                report['slices'].append(_sample(session, candidate, owners, segment, 'axial',
                                                 int(local_z + roi[0].start), box, device, growth))
        reports.append(_finish(report, owners, candidate))
    progress(100, 'Unreviewed MRI refinement candidates ready')
    return candidate, reports
