from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from .model import Box, Seed, Session, Volume


def fill_small_2d_holes(mask, spacing_yx, max_area_mm2=2500):
    holes, _ = ndi.label(ndi.binary_fill_holes(mask) & ~mask)
    sizes = np.bincount(holes.ravel())
    allowed = sizes * np.prod(spacing_yx) <= max_area_mm2
    allowed[0] = False
    return mask | allowed[holes]


def fill_bone_cavities(mask, spacing_zyx, available=None, progress=lambda *_:None, *, max_passes=6):
    """Fill bone interiors visible as enclosed gaps in any orthogonal plane.

    A canal open at a scan end is not a3Dhole. Orthogonal2Dfills recover such
    interiors without dilating the outer contour or overwriting another bone.
    """
    if not mask.any():
        return mask.copy()
    box = []
    for axis in range(3):
        occupied = np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != axis)))
        box.append(slice(max(0, int(occupied[0])-1), min(mask.shape[axis], int(occupied[-1])+2)))
    box = tuple(box)
    core = mask[box]
    allowed = np.ones(core.shape, bool) if available is None else available[box] | core
    work = ndi.binary_fill_holes(core) & allowed
    total = sum(core.shape)*max_passes; completed = 0
    for iteration in range(max_passes):
        before_count=int(work.sum())
        for axis in range(3):
            planes = np.moveaxis(work, axis, 0)
            permitted = np.moveaxis(allowed, axis, 0)
            spacing = tuple(value for i, value in enumerate(spacing_zyx) if i != axis)
            for index, plane in enumerate(planes):
                if index % 16 == 0:
                    progress(int((completed + index) / total * 95), '沿三平面循环填充骨内髓腔')
                planes[index] = fill_small_2d_holes(plane, spacing) & permitted[index]
            completed += len(planes)
        work=ndi.binary_fill_holes(work)&allowed
        if int(work.sum())==before_count:break
    progress(100, '骨内髓腔填充完成')
    result = mask.copy(); result[box] = work
    return result


def initial_segmentation(volume: Volume, roi: Box, threshold=200, upper=4000,
                         min_mm3=200, fill=True, progress=lambda *_: None, *, patient_mask=None, cleanup_report=None):
    if volume.is_mr:raise ValueError('MRI 信号不是 HU，请使用 MRI 专用分割流程。')
    if not np.isfinite([threshold, upper, min_mm3]).all() or threshold >= upper or min_mm3 < 0:
        raise ValueError('阈值上下限或最小体积无效。')
    data = volume.hu[roi]
    progress(8, '提取范围内的高密度结构')
    mask = (data >= threshold) & (data <= upper)
    if patient_mask is not None:
        if patient_mask.shape != data.shape or patient_mask.dtype != np.bool_:
            raise ValueError('人体轮廓与当前范围不一致。')
        from .preprocessing import exclude_external_attachments
        mask, attachment_report = exclude_external_attachments(mask, patient_mask, volume.spacing_zyx,
            lambda value,text:progress(8+int(value*.25),text))
        if cleanup_report is not None:cleanup_report.update(attachment_report)
    if fill:
        for z in range(mask.shape[0]):
            if z % 16 == 0:
                progress(35 + int(z / mask.shape[0] * 35), '填充骨内部')
            # A cast/skin-edge ring can enclose the entire leg. Filling that
            # opening would convert muscle into a solid "bone" component.
            mask[z] = fill_small_2d_holes(mask[z], volume.spacing_zyx[1:])
    if patient_mask is not None:
        mask &= patient_mask
    progress(75, '区分连通骨块')
    components, _ = ndi.label(mask, structure=ndi.generate_binary_structure(3, 3))
    counts = np.bincount(components.ravel())
    counts[0] = 0
    keep = np.flatnonzero(counts * volume.voxel_mm3 >= min_mm3)
    keep = keep[np.argsort(counts[keep])[::-1]]
    if len(keep) == 0:
        raise ValueError('当前范围与阈值没有找到骨块，请降低阈值或调整范围。')
    if len(keep) > 128:
        raise ValueError(f'检测到 {len(keep)} 个碎块，请提高最小骨块体积或收小范围。')
    lut = np.zeros(len(counts), dtype=np.uint16)
    lut[keep] = np.arange(1, len(keep) + 1)
    progress(90, '准备可编辑骨块')
    return lut[components], len(keep)


def install_initial(session: Session, result, proposals=None):
    local, n = result
    names = {entry['id']: entry for entry in proposals or []}
    lut = np.zeros(n + 1, dtype=np.uint16)
    for i in range(1, n + 1):
        proposal = names.get(i)
        segment = session.add_segment(proposal['name'] if proposal and proposal['name'] else None)
        if proposal:
            segment.recognition = {**proposal, 'id': segment.id}
        lut[i] = segment.id
    session.apply(session.roi, lut[local], '自动初分割')
    # Old seeds no longer describe the newly generated objects.
    session.seeds = [s for s in session.seeds if not inside(s.zyx, session.roi)]
    return lut[1:].tolist()


def inside(zyx, box):
    return all(s.start <= i < s.stop for i, s in zip(zyx, box))


def plane_box(shape, plane, index):
    axis = {'axial': 0, 'coronal': 1, 'sagittal': 2}[plane]
    box = [slice(0, n) for n in shape]
    box[axis] = slice(index, index + 1)
    return tuple(box)


def paint(session: Session, plane, center, radius_mm, ident, erase=False):
    if ident <= 0 or session.segment(ident) is None or radius_mm <= 0:
        return
    from .mr_policy import require_mri_modeling_source
    require_mri_modeling_source(session,plane)
    if not erase and session.segment(ident).presence=='absent':
        raise ValueError('该结构已标记缺失，请先修改存在状态再绘制。')
    if session._counts_version!=session.history.version:session._counts_cache=None
    fixed = {'axial': 0, 'coronal': 1, 'sagittal': 2}[plane]
    if not inside(center, session.roi):
        return
    box = []
    for axis, (c, sp, extent) in enumerate(zip(center, session.volume.spacing_zyx, session.roi)):
        r = 0 if axis == fixed else int(np.ceil(radius_mm / sp))
        box.append(slice(max(extent.start, c - r), min(extent.stop, c + r + 1)))
    grids = np.ogrid[tuple(slice(s.start, s.stop) for s in box)]
    distance = sum(((g - c) * sp) ** 2 for axis, (g, c, sp) in enumerate(zip(grids, center, session.volume.spacing_zyx)) if axis != fixed)
    view = session.labels[tuple(box)]
    target = distance <= radius_mm**2
    if erase:
        target = target & (view == ident)
    old=view[target].copy()
    view[target] = 0 if erase else ident
    session._adjust_counts(old,view[target])


def assign_component(session: Session, point, target_id,plane=None):
    from .mr_policy import require_mri_modeling_source
    require_mri_modeling_source(session,plane)
    if not inside(point, session.roi) or session.segment(target_id) is None:
        raise ValueError('请在当前范围内选择骨块。')
    if session.segment(target_id).presence=='absent':raise ValueError('缺失结构不能接收分割体素。')
    source = int(session.labels[point])
    if source == 0:
        raise ValueError('点击位置没有已分割骨块。')
    if source == target_id:
        return None
    local = session.labels[session.roi].copy()
    p = tuple(i - s.start for i, s in zip(point, session.roi))
    if plane is not None:
        axis={'axial':0,'coronal':1,'sagittal':2}[plane]
        index=[slice(None)]*3;index[axis]=p[axis]
        view=local[tuple(index)]
        cc,_=ndi.label(view==source,structure=ndi.generate_binary_structure(2,2))
        q=tuple(value for i,value in enumerate(p) if i!=axis)
        view[cc==cc[q]]=target_id
        return local
    cc, _ = ndi.label(local == source, structure=ndi.generate_binary_structure(3, 3))
    local[cc == cc[p]] = target_id
    return local


def split_from_seeds(session: Session, progress=lambda *_: None):
    from .mr_policy import require_mri_modeling_source
    require_mri_modeling_source(session)
    roi = session.roi
    local = session.labels[roi].copy()
    mask = local > 0
    markers = np.zeros(local.shape, dtype=np.int32)
    seeds = [s for s in session.seeds if inside(s.zyx, roi)]
    if any(session.segment(s.label) and session.segment(s.label).presence=='absent' for s in seeds):
        raise ValueError('种子包含已标记缺失的结构，请先修改状态或清除该种子。')
    if not seeds:
        raise ValueError('先选中目标骨标签，再在骨内部放置种子。每块待分离骨至少放一个种子。')
    for seed in seeds:
        p = tuple(i - s.start for i, s in zip(seed.zyx, roi))
        if not mask[p]:
            raise ValueError('有种子落在骨块外，请清除后重新放置。')
        if markers[p] and markers[p] != seed.label:
            raise ValueError('同一位置有不同骨的种子，请移开其中一个。')
        markers[p] = seed.label
    if len({s.label for s in seeds}) < 2:
        raise ValueError('分离需要至少两个不同骨标签的种子。')
    progress(15, '计算骨内距离场')
    distance = ndi.distance_transform_edt(mask, sampling=session.volume.spacing_zyx)
    progress(55, '根据种子分离相邻骨')
    result = watershed(-distance, markers=markers, mask=mask, connectivity=ndi.generate_binary_structure(3, 3))
    # Unseeded disconnected components retain their current labels.
    seeded = result > 0
    local[seeded] = result[seeded].astype(np.uint16)
    progress(100, '种子分离完成，请核对关节面')
    return local


def clean_segment(session: Session, ident, operation, progress=lambda *_:None):
    from .mr_policy import require_mri_modeling_source,sagittal_only
    require_mri_modeling_source(session)
    local = session.labels[session.roi].copy()
    mask = local == ident
    if not mask.any():
        raise ValueError('当前骨标签为空。')
    if operation == 'largest':
        cc, _ = ndi.label(mask, structure=ndi.generate_binary_structure(3, 3))
        counts = np.bincount(cc.ravel()); counts[0] = 0
        local[mask & (cc != counts.argmax())] = 0
    elif operation == 'fill':
        if sagittal_only(session):
            # Complete only acquired sagittal contours, without consulting
            # axial/coronal reformats or filling across acquisition gaps.
            filled=mask.copy()
            for z,plane in enumerate(mask):
                progress(int(100*z/len(mask)),'沿原始矢状采集层填充骨内轮廓')
                filled[z]=fill_small_2d_holes(plane,session.volume.spacing_zyx[1:])
        else:filled = fill_bone_cavities(mask, session.volume.spacing_zyx, local == 0, progress)
        local[filled & (local == 0)] = ident
    elif operation == 'clear':
        local[mask] = 0
    else:
        raise ValueError('未知编辑操作。')
    return local
