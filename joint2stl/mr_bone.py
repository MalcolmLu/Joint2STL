"""Local SegmentAnyBone proposals on the unchanged native MR sampling grid.

The public API returns a private ROI candidate and metadata, never commits a
Session. Ligament and unregistered voxel owners remain protected. Bone names
are conservative geometry candidates, not learned anatomical classifications.
"""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from . import ai
from .model import PALETTE, Segment
from .mr_policy import require_mri_modeling_source,modeling_source_metadata


CHECKPOINT_BYTES = 46_221_025
CHECKPOINT_SHA256 = '43209385a63290f7a10bb80eb1bbb47186abd64c73f507b3fcae6d11542e1d49'
CHECKPOINT_URL = ('https://raw.githubusercontent.com/mazurowski-lab/SlicerSegmentHumanBody/'
                  '5f026279291d2980a01274a0d0aaa6c1a4d68960/SegmentHumanBody/models/bone_sam.pth')
IMAGE_SIZE = 1024
PIXEL_MEAN = (123.675, 116.28, 103.53)
PIXEL_STD = (58.395, 57.12, 57.375)
_BONE_STRUCTURES = ('femur', 'tibia', 'fibula', 'patella')
CORE_RADIUS_MM = 1.5
CORE_MIN_MM3 = 75.


def checkpoint_path():
    return Path(__file__).resolve().parents[1] / 'models' / 'bone_sam.pth'


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def mri_bone_model_ready(*, verify=False):
    """Cheap installation check; loading always verifies the complete hash."""
    path = checkpoint_path()
    try:
        return (path.is_file() and path.stat().st_size == CHECKPOINT_BYTES
                and importlib.util.find_spec('torch') is not None
                and importlib.util.find_spec('timm') is not None
                and (not verify or _sha256(path) == CHECKPOINT_SHA256))
    except OSError:
        return False


@lru_cache(maxsize=2)
def _load_model_on_device(device):
    import torch
    from .vendor.segment_any_bone.build import build_bone_model

    path = checkpoint_path()
    if not path.is_file():
        raise ValueError('SegmentAnyBone is not installed; run scripts/setup_mri_model.py.')
    if path.stat().st_size != CHECKPOINT_BYTES or _sha256(path) != CHECKPOINT_SHA256:
        raise ValueError('SegmentAnyBone checkpoint size or SHA-256 verification failed.')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if device == 'mps':
        torch.mps.set_per_process_memory_fraction(.35)
    elif device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.5)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                              for k, v in state.items()):
        raise ValueError('SegmentAnyBone must contain a plain tensor state dictionary.')
    model = build_bone_model()
    model.load_state_dict(state, strict=True)
    # Loading before eval is essential: eval refreshes nonpersistent bias caches.
    model.to(device).eval()
    return model, device


def load_model(device='auto'):
    if device not in ('auto', 'mps', 'cuda', 'cpu'):
        raise ValueError('MRI bone device must be auto, mps, cuda, or cpu.')
    return _load_model_on_device(ai.torch_device(device))


def _minmax_rgb(raw):
    """Match Slicer scalar -> min/max uint8 RGB, sanitizing a private copy."""
    raw = np.asarray(raw)
    if raw.ndim != 2 or min(raw.shape) < 1 or raw.dtype.kind not in 'uif':
        raise ValueError('A native scalar MR slice is required.')
    finite = np.isfinite(raw)
    count = int(finite.sum())
    low, high = (float(raw[finite].min()), float(raw[finite].max())) if count else (0., 0.)
    valid = count > 0 and high > low
    gray = np.zeros(raw.shape, dtype=np.uint8)
    if valid:
        spread = high - low
        if not np.isfinite(spread):
            raise ValueError('MR slice signal range is outside the inference numeric range.')
        values = np.asarray(raw, dtype=np.float64)
        values = np.where(finite, values, low)
        gray = np.clip((values - low) / spread * 255., 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1), {
        'method': 'native_slice_finite_minmax_uint8_rgb', 'minimum': low, 'maximum': high,
        'nonfinite_pixels_replaced': int(raw.size - count), 'degenerate': not valid,
        'original_size_yx': list(raw.shape),
    }


def _preprocess_rgb(rgb, device):
    """SAM's PIL longest-side resize, normalization, and zero-normalized pad.

    Pillow is used intentionally: SAM's original apply_image uses PIL bilinear
    uint8 resampling. Float-only Torch resizing changes those quantized pixels.
    No torchvision transforms, cv2, matplotlib, or CT preprocessing is imported.
    """
    import torch
    from torch.nn import functional as F
    from PIL import Image

    height, width = rgb.shape[:2]
    scale = IMAGE_SIZE / max(height, width)
    nh, nw = int(height * scale + .5), int(width * scale + .5)
    resized = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.Resampling.BILINEAR))
    tensor = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).float().to(device)
    mean = torch.tensor(PIXEL_MEAN, dtype=torch.float32, device=device)[:, None, None]
    std = torch.tensor(PIXEL_STD, dtype=torch.float32, device=device)[:, None, None]
    tensor = F.pad((tensor - mean) / std, (0, IMAGE_SIZE - nw, 0, IMAGE_SIZE - nh))
    return tensor[None], (nh, nw)


def _postprocess_bone(logits, input_size, original_size):
    from torch.nn import functional as F
    if logits.ndim != 4 or logits.shape[:2] != (1, 2):
        raise ValueError('SegmentAnyBone must return its two trained semantic masks.')
    # mask_decoder(multimask_output=True) has already sliced SAM token 0 away.
    # The official trained bone output is masks[1], regardless of IoU scores.
    bone = F.interpolate(logits[:, 1:2], (IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
    bone = bone[..., :input_size[0], :input_size[1]]
    bone = F.interpolate(bone, original_size, mode='bilinear', align_corners=False)
    return bone[0, 0]


def _infer_rgb(rgb, device):
    import torch
    model, actual = load_model(device)
    with torch.inference_mode():
        tensor, input_size = _preprocess_rgb(rgb, actual)
        embedding = model.image_encoder(tensor)
        sparse, dense = model.prompt_encoder(points=None, boxes=None, masks=None)
        logits, _scores = model.mask_decoder(
            image_embeddings=embedding, image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=True)
        bone = _postprocess_bone(logits, input_size, rgb.shape[:2])
        if not bool(torch.isfinite(bone).all().item()):
            raise ValueError('SegmentAnyBone returned nonfinite mask logits.')
        mask = (bone > 0).cpu().numpy()
    return mask, actual, list(input_size)


def _is_mps_operation_failure(exc):
    return (isinstance(exc, (RuntimeError, NotImplementedError))
            and any(word in str(exc).lower() for word in ('mps', 'metal', 'mpsgraph')))


def infer_bone_slice(raw, device='auto'):
    """Binary native-plane prediction; CPU retry only after an actual MPS error."""
    rgb, normalization = _minmax_rgb(raw)
    details = {'normalization': normalization, 'device': 'not_run', 'inference_performed': False}
    if normalization['degenerate']:
        return np.zeros(rgb.shape[:2], dtype=bool), details
    if device not in ('auto', 'mps', 'cuda', 'cpu'):
        raise ValueError('MRI bone device must be auto, mps, cuda, or cpu.')
    requested = ai.torch_device(device)
    failure = None
    try:
        mask, actual, resized = _infer_rgb(rgb, requested)
    except (RuntimeError, NotImplementedError) as exc:
        if requested != 'mps' or not _is_mps_operation_failure(exc):
            raise
        failure = str(exc)[:1500]
    if failure is not None:
        # Drop a failed MPS model and temporary exception tensors before retry.
        _load_model_on_device.cache_clear()
        import torch
        try:
            torch.mps.empty_cache()
        except RuntimeError:
            # A broken MPS allocator must not prevent the explicit CPU retry.
            pass
        mask, actual, resized = _infer_rgb(rgb, 'cpu')
        details['mps_fallback_reason'] = failure
    details.update(device=actual, resized_size_yx=resized, inference_performed=True)
    return mask, details


def _bone_records(session):
    """Resolve structural IDs without changing templates or allocating labels."""
    seen = set()
    for segment in session.segments:
        if (not isinstance(segment.id, (int, np.integer)) or isinstance(segment.id, (bool, np.bool_))
                or not 1 <= segment.id <= 65535 or segment.id in seen):
            raise ValueError('MRI segment IDs must be unique nonzero uint16 values.')
        seen.add(segment.id)
    protected = {s.id for s in session.segments
                 if s.kind != 'bone' or s.structure in ('acl', 'pcl')}
    bones = {s.id: s for s in session.segments if s.kind == 'bone' and s.id not in protected}
    templates = {}
    for key in _BONE_STRUCTURES:
        matches = [s for s in bones.values() if s.structure == key]
        if len(matches) != 1:
            raise ValueError('MRI knee templates are required; call ensure_mr_segments before generating bones.')
        templates[key] = matches[0]
    return bones, templates


def _separate_bone_instances(foreground, spacing_zyx, voxel_mm3):
    """Partition only existing foreground using meaningful physical cores.

    Six-connected eroded cores are markers, rather than individual distance
    peaks. Each original component without a sizeable core gets exactly one
    fallback marker at its strongest inside distance. Watershed runs with 26
    connectivity inside EACH original component, preventing cross-component
    label theft through diagonal contact. A one-marker component is unchanged.
    """
    from skimage.segmentation import watershed

    raw, raw_count = ndi.label(foreground, structure=ndi.generate_binary_structure(3, 1))
    # Explicit zero border makes truncated ROI faces symmetric, including an
    # all-foreground ROI. Distances are measured on the native anisotropic grid.
    distance = ndi.distance_transform_edt(np.pad(foreground, 1), sampling=spacing_zyx)[1:-1, 1:-1, 1:-1]
    cores, core_count = ndi.label(distance > CORE_RADIUS_MM,
                                 structure=ndi.generate_binary_structure(3, 1))
    core_sizes = np.bincount(cores.ravel(), minlength=core_count + 1)
    keep = np.flatnonzero(core_sizes * voxel_mm3 + 1e-9 >= CORE_MIN_MM3)
    keep = keep[keep != 0]
    core_lut = np.zeros(core_count + 1, dtype=np.int32)
    core_lut[keep] = np.arange(1, len(keep) + 1)
    markers = core_lut[cores]
    marker_sources = []
    # A six-connected core is a subset of exactly one original raw component.
    core_owners = ndi.maximum(raw, labels=cores, index=keep).astype(int) if len(keep) else []
    owners = [0]
    for marker, (core, owner) in enumerate(zip(keep, core_owners), 1):
        owners.append(int(owner))
        marker_sources.append({'marker': marker, 'raw_component': int(owner),
                               'source': 'eroded_core', 'voxels': int(core_sizes[core]),
                               'mm3': float(core_sizes[core] * voxel_mm3)})
    represented = set(owners[1:])
    missing = [i for i in range(1, raw_count + 1) if i not in represented]
    positions = ndi.maximum_position(distance, labels=raw, index=missing) if missing else []
    for owner, position in zip(missing, positions):
        marker = len(owners)
        owners.append(owner)
        markers[position] = marker
        marker_sources.append({'marker': marker, 'raw_component': owner,
                               'source': 'strongest_inside_fallback', 'voxels': 1,
                               'mm3': float(voxel_mm3),
                               'maximum_inside_distance_mm': float(distance[position])})
    instances = np.zeros_like(raw)
    markers_per_raw = np.bincount(owners[1:], minlength=raw_count + 1)
    owner_single_marker = np.zeros(raw_count + 1, dtype=np.int32)
    for marker, owner in enumerate(owners[1:], 1):
        owner_single_marker[owner] = marker
    split_raw_components = 0
    for owner, box in enumerate(ndi.find_objects(raw), 1):
        if box is None:
            continue
        mask = raw[box] == owner
        view = instances[box]
        if markers_per_raw[owner] == 1:
            view[mask] = owner_single_marker[owner]
        else:
            split_raw_components += 1
            local_markers = np.where(mask, markers[box], 0)
            separated = watershed(-distance[box], local_markers, mask=mask,
                                  connectivity=ndi.generate_binary_structure(3, 3),
                                  watershed_line=True)
            view[mask] = separated[mask]
    seam_voxels = int(np.count_nonzero(foreground & (instances == 0)))
    report = {
        'method': 'physical_core_marker_distance_watershed_v1',
        'core_radius_mm': CORE_RADIUS_MM, 'core_min_mm3': CORE_MIN_MM3,
        'raw_component_connectivity': 6, 'core_connectivity': 6,
        'watershed_connectivity': 26, 'watershed_line': True,
        'watershed_scope': 'each original connected foreground component',
        'raw_component_count': int(raw_count), 'eroded_core_count': int(core_count),
        'core_marker_count': len(keep), 'fallback_marker_count': len(missing),
        'markers_total': len(owners) - 1, 'marker_sources': marker_sources,
        'raw_components_split': split_raw_components,
        'raw_foreground_voxels': int(foreground.sum()),
        'watershed_retained_voxels': int(np.count_nonzero(instances)),
        'seam_removed_voxels': seam_voxels, 'outside_foreground_added_voxels': 0,
    }
    return instances, report


def _component_rows(components, sizes, session):
    """Measure retained native components directly in patient LPS millimetres."""
    rows = []
    origin = np.array([r.start for r in session.roi])
    for ident, box in enumerate(ndi.find_objects(components), 1):
        if box is None or sizes[ident] == 0:
            continue
        coords = np.column_stack(np.nonzero(components[box] == ident))
        coords += origin + np.array([r.start for r in box])
        world = session.volume.world(coords)
        quantiles = np.percentile(world, [5, 50, 95], axis=0)
        centred = world - world.mean(axis=0)
        covariance = centred.T @ centred / max(1, len(world))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, -1]
        rows.append({
            'component': ident, 'voxels': int(sizes[ident]),
            'mm3': float(sizes[ident] * session.volume.voxel_mm3),
            'centroid_zyx': coords.mean(axis=0).tolist(),
            'centroid_lps_mm': world.mean(axis=0).tolist(),
            'bounds_lps_mm': [world.min(axis=0).tolist(), world.max(axis=0).tolist()],
            'quantiles_lps_mm_5_50_95': quantiles.tolist(),
            'robust_extents_lps_mm': (quantiles[2] - quantiles[0]).tolist(),
            'principal_axis_lps': axis.tolist(),
            'principal_variance_ratio': float(eigenvalues[-1] / max(1e-9, eigenvalues[-2])),
            'name_basis': [], 'reviewed': False,
        })
    return sorted(rows, key=lambda r: (-r['voxels'], r['component']))


def _geometry_names(rows, session):
    """Name only an unambiguous long-bone pair and unique small companions.

    S=+LPS z, anterior=-LPS y. Fibula needs a unique lower lateral rod;
    unknown laterality keeps the fibular name unresolved without inferring L/R.
    Unknown, fused, truncated and extra objects remain separate unresolved IDs.
    """
    assigned = {}
    max_spacing = float(max(session.volume.spacing_zyx))

    def long_bone(row):
        left, posterior, superior = row['robust_extents_lps_mm']
        return (superior >= max(15., 3 * max_spacing)
                # Knee-only fields can truncate a shaft to its condylar width.
                and superior >= max(left, posterior, 1.)
                and abs(row['principal_axis_lps'][2]) >= .75
                and row['principal_variance_ratio'] >= 1.25)

    def main_joint_bone(row):
        # Knee-only fields often truncate both shafts. A broad metaphysis can
        # have its largest PCA axis across L/R, even with a clear proximal and
        # distal pair. Use physical extent/dominance and pair alignment here;
        # keep the stricter elongated test for a small fibula candidate.
        left,posterior,superior=row['robust_extents_lps_mm']
        return superior>=max(15.,3*max_spacing) and superior>=.5*max(left,posterior,1.)

    if len(rows) < 2:
        return assigned, ['Fewer than two distinct sizeable components; named bones unresolved.']
    first, second = rows[:2]
    if (not main_joint_bone(first) or not main_joint_bone(second)
            or second['voxels'] < .15 * first['voxels']
            or (len(rows) > 2 and rows[2]['voxels'] > .55 * second['voxels'])):
        return assigned, ['Largest components do not form a clear isolated long-bone pair.']
    high, low = sorted((first, second), key=lambda r: r['centroid_lps_mm'][2], reverse=True)
    hc, lc = np.asarray(high['centroid_lps_mm']), np.asarray(low['centroid_lps_mm'])
    he, le = np.asarray(high['robust_extents_lps_mm']), np.asarray(low['robust_extents_lps_mm'])
    hq, lq = np.asarray(high['quantiles_lps_mm_5_50_95']), np.asarray(low['quantiles_lps_mm_5_50_95'])
    clear = (hc[2] - lc[2] >= max(12., .2 * min(he[2], le[2]))
             and abs(hq[0, 2]-lq[2, 2])<=35.
             and abs(hc[0] - lc[0]) <= max(20., .45 * (he[0] + le[0]))
             and abs(hc[1] - lc[1]) <= max(25., .6 * (he[1] + le[1])))
    if not clear:
        return assigned, ['Physical S separation/alignment is ambiguous; named bones unresolved.']
    assigned[high['component']] = 'femur'
    assigned[low['component']] = 'tibia'
    high['name_basis'] = ['clear largest long-bone pair', 'higher centroid in patient +S']
    low['name_basis'] = ['clear largest long-bone pair', 'lower centroid in patient +S']
    joint_s = float((hq[0, 2] + lq[2, 2]) / 2)
    centre = (hc + lc) / 2
    patella = []
    for row in rows[2:]:
        c = np.asarray(row['centroid_lps_mm'])
        extent = np.asarray(row['robust_extents_lps_mm'])
        if (max(100., .01 * low['mm3']) <= row['mm3'] <= .4 * low['mm3']
                and extent[0] >= 4. and extent[1] >= 3. and extent[2] >= 5.
                and c[1] < min(hc[1], lc[1]) - max(5., .2 * max(he[1], le[1]))
                and abs(c[0] - centre[0]) <= max(10., .5 * max(he[0], le[0]))
                and abs(c[2] - joint_s) <= min(40., max(25., .25 * (he[2] + le[2])))
                and max(extent) <= max(40., .7 * min(he[2], le[2]))):
            patella.append(row)
    if len(patella) == 1:
        row = patella[0]
        assigned[row['component']] = 'patella'
        row['name_basis'] = ['unique small anterior component near the physical joint level']

    side = str(session.side).strip().lower()
    lateral_sign = 1 if side in ('左', '左侧', 'left', 'l') else -1 if side in ('右', '右侧', 'right', 'r') else 0
    fibula = []
    for row in rows[2:]:
        if row['component'] in assigned:
            continue
        c = np.asarray(row['centroid_lps_mm'])
        q = np.asarray(row['quantiles_lps_mm_5_50_95'])
        lateral_offset = lateral_sign * (c[0] - lc[0]) if lateral_sign else abs(c[0] - lc[0])
        if (row['voxels'] <= .3 * low['voxels'] and long_bone(row)
                and lateral_offset >= max(4., .15 * le[0])
                and c[1] >= lc[1]
                and q[2, 2] <= lq[2, 2] + 5.
                and q[0, 2] <= lq[1, 2]):
            fibula.append(row)
    if len(fibula) == 1 and lateral_sign:
        row = fibula[0]
        assigned[row['component']] = 'fibula'
        row['name_basis'] = ['unique small lower long component lateral/posterior to tibia',
                             'explicit patient laterality' if lateral_sign else
                             'absolute lateral offset; patient laterality remains unknown']
    limitations = ['Names are physical geometry candidates, not trained anatomical classifications.']
    if not lateral_sign:
        limitations.append('Patient laterality is not inferred; fibular candidates remain unresolved until laterality and anatomy are reviewed.')
    return assigned, limitations


def prepare_mri_bones(session, progress=lambda *_: None, device='auto'):
    """Return (private ROI uint16 candidate, report, bone metadata dicts).

    Existing structural IDs stay fixed. Only registered bone owners are cleared
    for regeneration; ligament, other kinds and orphan voxel IDs are reserved.
    No Session metadata/history, original image values, or affine is modified.
    """
    if session.volume.modality != 'MR' or session.joint != 'knee':
        raise ValueError('SegmentAnyBone integration supports MR knee volumes only.')
    require_mri_modeling_source(session)
    if device not in ('auto', 'mps', 'cuda', 'cpu'):
        raise ValueError('MRI bone device must be auto, mps, cuda, or cpu.')
    bones, templates = _bone_records(session)
    min_mm3 = session.parameters.get('min_mm3', 200.)
    if isinstance(min_mm3, (bool, np.bool_)) or not np.isfinite(float(min_mm3)) or float(min_mm3) < 0:
        raise ValueError('MRI minimum component volume must be finite nonnegative mm3.')
    min_mm3 = float(min_mm3)
    roi = session.roi
    owners = session.labels[roi].copy()
    bone_ids = list(bones)
    old_bone = np.isin(owners, bone_ids)
    editable = (owners == 0) | old_bone
    candidate = owners.copy()
    candidate[old_bone] = 0
    prediction = np.zeros(owners.shape, dtype=bool)
    entries, fallbacks, devices = [], [], []
    run_device = device
    progress(0, 'Preparing native MRI bone candidate')
    for local_z, z in enumerate(range(roi[0].start, roi[0].stop)):
        progress(5 + int(75 * local_z / len(prediction)),
                 f'SegmentAnyBone native slice {local_z + 1}/{len(prediction)}')
        mask, detail = infer_bone_slice(session.volume.values[z], run_device)
        mask = np.asarray(mask)
        if mask.shape != session.volume.values.shape[1:] or mask.dtype != bool:
            raise ValueError('MRI bone model returned an invalid native-plane binary mask.')
        prediction[local_z] = mask[roi[1:]] & editable[local_z]
        detail = dict(detail)
        actual = detail.get('device', 'not_run')
        if actual != 'not_run' and actual not in devices:
            devices.append(actual)
        if 'mps_fallback_reason' in detail:
            fallbacks.append({'slice_index_0based': z, 'reason': detail['mps_fallback_reason']})
            run_device = 'cpu'
        entries.append({'slice_1based': z + 1, 'fixed_index_0based': z,
                        'fixed_axis': 0, 'native_acquisition_plane': True,
                        'proposed_bone_voxels_in_roi': int(prediction[local_z].sum()), **detail})

    progress(82, 'Separating native 3D bone components in physical millimetres')
    components, instance_report = _separate_bone_instances(
        prediction, session.volume.spacing_zyx, session.volume.voxel_mm3)
    count = instance_report['markers_total']
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    kept = np.flatnonzero(sizes * session.volume.voxel_mm3 + 1e-9 >= min_mm3)
    kept = kept[kept != 0]
    accepted = np.zeros(count + 1, dtype=bool)
    accepted[kept] = True
    components[~accepted[components]] = 0
    retained_sizes = sizes.copy()
    retained_sizes[~accepted] = 0
    instance_report['min_instance_mm3'] = min_mm3
    instance_report['filtered_small_instance_voxels'] = int(sizes[1:][~accepted[1:]].sum())
    instance_report['retained_foreground_voxels'] = int(np.count_nonzero(components))
    rows = _component_rows(components, retained_sizes, session)
    names, geometry_limits = _geometry_names(rows, session)
    metadata = {ident: asdict(segment) for ident, segment in bones.items()}
    next_id = max(session.next_segment_id(), int(session.labels.max()) + 1)
    label_lut = np.zeros(count + 1, dtype=np.uint16)
    unresolved_number = 0
    for row in rows:
        key = names.get(row['component'])
        if key and templates[key].presence == 'absent':
            geometry_limits.append(f'{key} template is marked absent; component kept unresolved.')
            key = None
        if key:
            ident = templates[key].id
        else:
            if next_id > 65535:
                raise ValueError('No uint16 IDs remain for unresolved MRI bone components.')
            ident, next_id = next_id, next_id + 1
            unresolved_number += 1
            segment = Segment(ident, f'MRI 骨块 {unresolved_number:02d} · 待命名',
                              PALETTE[(ident - 1) % len(PALETTE)], kind='bone', structure='unresolved')
            metadata[ident] = asdict(segment)
            row['name_basis'] = ['anatomy unresolved; component kept separate']
        row.update(label_id=int(ident), structure=key or 'unresolved', label_name=metadata[ident]['name'])
        label_lut[row['component']] = ident
    generated = label_lut[components]
    candidate[generated != 0] = generated[generated != 0]
    bone_rows = []
    for ident, record in metadata.items():
        after = candidate == ident
        associated = [r['component'] for r in rows if r['label_id'] == ident]
        recognition = {
            'method': 'mr_segmentanybone_geometry_candidate_v2', 'reviewed': False,
            'structure': record['structure'], 'component_ids': associated,
            'model_semantics': 'binary_bone_only', 'voxel_count': int(after.sum()),
            'manual_name_preserved': bool(bones.get(ident) and bones[ident].reviewed),
            'needs_review': True,
        }
        record['reviewed'] = False
        record['recognition'] = recognition
        bone_rows.append({'id': int(ident), 'name': record['name'], 'kind': 'bone',
                          'structure': record['structure'], 'reviewed': False,
                          'voxels': int(after.sum()), 'mm3': float(after.sum() * session.volume.voxel_mm3),
                          'components': associated,
                          'added_voxels': int(np.count_nonzero(after & (owners != ident))),
                          'removed_voxels': int(np.count_nonzero((owners == ident) & ~after))})
    generated_bone = generated != 0
    component_counts = [dict(component=i, voxels=int(sizes[i]),
                             mm3=float(sizes[i] * session.volume.voxel_mm3),
                             retained=bool(accepted[i])) for i in range(1, count + 1)]
    report = {
        'method': 'mr_segmentanybone_native_slices_v2', 'model': 'SegmentAnyBone',
        'modality': 'MR', 'joint': 'knee', 'reviewed': False,
        'checkpoint_sha256': CHECKPOINT_SHA256, 'checkpoint_source': CHECKPOINT_URL,
        'license': 'CC BY-NC 4.0', 'precision': 'float32; CUDA TF32 disabled',
        'torch_threads_max': 4, 'per_process_memory_fractions': {'cuda': .5, 'mps': .35},
        'requested_device': device, 'device': devices[-1] if devices else 'not_run',
        'devices_used': devices, 'mps_cpu_fallbacks': fallbacks,
        'inference_performed': any(e.get('inference_performed', False) for e in entries),
        'native_acquisition_plane': True, 'fixed_axis': 0,
        'modeling_source':modeling_source_metadata(session),
        'roi_zyx_0based': [[r.start, r.stop] for r in roi],
        'intensity_units': 'MR signal', 'normalization': 'per_native_slice_finite_minmax_uint8_rgb',
        'model_output': 'multimask_output=True; second returned mask (masks[1]) > 0',
        'depth_attention': False, 'outer_dilation_mm': 0.,
        'min_component_mm3': min_mm3, 'voxel_mm3': float(session.volume.voxel_mm3),
        'connectivity': 26, 'raw_component_connectivity': 6,
        'instance_separation': instance_report,
        'core_radius_mm': CORE_RADIUS_MM, 'core_min_mm3': CORE_MIN_MM3,
        'markers_total': instance_report['markers_total'],
        'seam_removed_voxels': instance_report['seam_removed_voxels'],
        'raw_foreground_voxels': instance_report['raw_foreground_voxels'],
        'retained_foreground_voxels': instance_report['retained_foreground_voxels'],
        'components_before_filter': int(count),
        'components_retained': len(rows), 'component_counts': component_counts,
        'components': rows, 'bones': bone_rows, 'slices': entries,
        'added_voxels': int(np.count_nonzero(generated_bone & (owners == 0))),
        'removed_voxels': int(np.count_nonzero(old_bone & ~generated_bone)),
        'relabeled_bone_voxels': int(np.count_nonzero(old_bone & generated_bone & (owners != candidate))),
        'preserved_nonbone_voxels': int(np.count_nonzero((owners != 0) & ~old_bone)),
        'new_ligament_voxels': 0,
        'limitations': [
            'No ground-truth comparison; binary MRI bone proposals require native-image review.',
            'The single-checkpoint Slicer path omits the separate VNet depth-attention model.',
            'ACL/PCL are not inferred; existing ligament/foreign ownership is preserved.',
            'No hole filling, outer dilation, template completion, or forced component fusion.',
            *geometry_limits,
        ],
    }
    progress(100, 'Unreviewed MRI bone candidates ready')
    return candidate, report, list(metadata.values())
