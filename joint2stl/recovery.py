"""Add CT-supported missing bone edges using competing existing bone markers."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from .preprocessing import patient_region
from .segmentation import fill_bone_cavities
from .gap_guard import protected_gap_mask, trim_new_contacts

RECOVERY_CONNECT_HU = 150
RECOVERY_SEED_HU = 200
RECOVERY_DISTANCE_MM = 5.0


def recover_ct_edges(session, ids, progress=lambda *_:None, *, labels=None, patient_mask=None, max_distance_mm=RECOVERY_DISTANCE_MM,
                     respect_reviewed_gaps=True, original_owners=None):
    """Return an additive candidate; never overwrite a currently labeled voxel.

    CT cortex can lie beyond the model's1mm uncertain-growth band. All existing
    bone markers compete inside CT-supported components before interiors fill,
    so recovery can follow a missing cortex without assigning the neighbor's
    cortex to this bone. Unseeded external structures do not join the candidate.
    """
    if session.volume.is_mr:raise ValueError('CT 骨缘补全使用 HU，不能用于 MRI；请使用 MRI 模型框选或画笔。')
    if not np.isfinite(max_distance_mm) or not 0 < max_distance_mm <= 12:
        raise ValueError('CT 骨缘补全距离无效。')
    roi = session.roi
    baseline = session.labels[roi] if labels is None else labels
    shape = tuple(s.stop-s.start for s in roi)
    if baseline.shape != shape or baseline.dtype != np.uint16:
        raise ValueError('CT 骨缘补全与选区不一致。')
    if original_owners is not None and (original_owners.shape != shape or original_owners.dtype != np.uint16):
        raise ValueError('原始标签归属与补全选区不一致。')
    ids = list(dict.fromkeys(int(i) for i in ids))
    objects = ndi.find_objects(baseline)
    ids = [i for i in ids if session.segment(i) is not None and 0 < i <= len(objects) and objects[i-1] is not None]
    if not ids:
        raise ValueError('请先选择非空的骨标签。')
    if patient_mask is None:
        try:
            patient_mask, outline_report = patient_region(session.volume, roi,
                lambda value,text:progress(int(value*.25),text))
        except ValueError as exc:
            progress(100,'人体轮廓依据不足，保留当前分割')
            return baseline.copy(),{'method':'competitive_ct_edge_recovery_v2','added_voxels':0,
                'existing_labels_preserved':True,'bones':[],'flags':[str(exc)]}
    else:
        outline_report = None
    if patient_mask.shape != shape or patient_mask.dtype != np.bool_:
        raise ValueError('CT 骨缘补全的人体范围无效。')
    hu = session.volume.hu[roi]
    spacing = session.volume.spacing_zyx
    protected = protected_gap_mask(session) if respect_reviewed_gaps else None
    if protected is not None:
        # Constrain the search itself, not only the output: do not grow through
        # a protected joint gap to reach CT voxels on its other side.
        patient_mask = patient_mask & ~protected
    result = baseline.copy()
    best_distance = np.full(shape, np.inf, np.float32)
    rows = []
    for index, ident in enumerate(ids):
        progress(25+int(index/len(ids)*70), f'沿 CT 皮质补全：{session.segment(ident).name}')
        margin = np.ceil(12/spacing).astype(int)
        box = tuple(slice(max(0,b.start-int(p)),min(n,b.stop+int(p)))
                    for b,p,n in zip(objects[ident-1],margin,shape))
        local = baseline[box]; current = local == ident
        values = hu[box]; inside = patient_mask[box]
        dense = (values >= RECOVERY_CONNECT_HU) & (values <= session.parameters.get('upper',4000)) & inside
        reserved=None
        if original_owners is not None:
            owner=original_owners[box]
            reserved=(owner!=0)&(owner!=ident)&(local==0)
            dense &= ~reserved
        strong = current & (values >= RECOVERY_SEED_HU) & inside
        if np.count_nonzero(strong)*session.volume.voxel_mm3 < 10:
            rows.append({'id':ident,'name':session.segment(ident).name,'added_voxels':0,
                         'flags':['CT 骨性依据不足，已保留当前分割']})
            continue
        markers = np.where(dense, local, 0).astype(np.int32)
        distances = ndi.distance_transform_edt(dense, sampling=spacing)
        owners = watershed(-distances,markers=markers,mask=dense,
                           connectivity=ndi.generate_binary_structure(3,3))
        core = owners == ident
        available = inside & ((local == 0) | current) & ((owners == 0) | core)
        if reserved is not None:available &= ~reserved
        envelope = fill_bone_cavities(core,spacing,available)
        distance_from_prior = ndi.distance_transform_edt(~current,sampling=spacing)
        cortex_recovery = envelope & available & (distance_from_prior <= max_distance_mm)
        # The outward recovery band must not also cap the depth of an enclosed
        # medullary cavity. Complete the interior after recovering its rim.
        completed_interior=fill_bone_cavities(current|cortex_recovery,spacing,available)
        additions = completed_interior & available & (local == 0)
        target = result[box]; best = best_distance[box]
        # Resolve any overlapping interior proposals by physical distance to the
        # original labels, keeping the order of processing from affecting ties.
        closer = additions & ((distance_from_prior < best-1e-6)
                 | ((np.abs(distance_from_prior-best)<=1e-6) & (ident < target)))
        target[closer] = ident; best[closer] = distance_from_prior[closer]
        rows.append({'id':ident,'name':session.segment(ident).name,
                     'ct_anchor_voxels':int(strong.sum()),'flags':[]})
    trimmed=trim_new_contacts(session,result,baseline=baseline) if respect_reviewed_gaps else 0
    for row in rows:
        ident = row['id']
        added = (result == ident) & (baseline == 0)
        count = int(added.sum());row['added_voxels'] = count
        row['added_dense_voxels'] = int(np.count_nonzero(added & (hu >= RECOVERY_SEED_HU)))
        prior_count = int(np.count_nonzero(baseline == ident))
        if count > prior_count*.2:row['flags'].append('补全范围较大，请对照 CT 核对')
    report = {'method':'competitive_ct_edge_recovery_v2','connect_hu':RECOVERY_CONNECT_HU,
              'seed_hu':RECOVERY_SEED_HU,'max_distance_mm':max_distance_mm,
              'distance_limit_scope':'outward CT-edge recovery; enclosed interiors filled separately',
              'roi_zyx':[[b.start,b.stop] for b in roi],
              'added_voxels':int(np.count_nonzero((result != baseline) & (baseline == 0))),
              'existing_labels_preserved':bool(np.array_equal(result[baseline>0],baseline[baseline>0])),
              'bones':rows,'outline':outline_report,
              'limitation':'CT-supported candidate; no expert reference segmentation.'}
    if protected is not None:
        report['protected_reviewed_gap_voxels'] = int(np.count_nonzero(protected))
    if trimmed:report['reviewed_gap_contact_proposals_rejected']=trimmed
    progress(100,'CT 骨缘补全完成；现有分割已保留')
    return result, report
