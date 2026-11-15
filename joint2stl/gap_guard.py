"""Keep a manually reviewed, explicitly recorded joint gap through automatic edits."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


METHOD = 'right_proximal_tibiofibular_seam_review_v1'


def _reviewed_gaps(session, roi, *, margin):
    """Read recorded background in the reviewed gap box. Ordinary cases allocate nothing.

    Old gap-review projects record the actual cut's bounding box but not a voxel
    list. Protect the zero-voxel subset within that small box. Trust it only
    while both of the named reviewed component records still carry their
    original provenance and the two labels remain present under these IDs.
    """
    for report in session.repair_reports:
        if not isinstance(report,dict) or report.get('method') != METHOD:
            continue
        if not report.get('segmented_bones_do_not_touch_26_neighborhood'):
            continue
        ids, box = report.get('label_ids'), report.get('seam_local_edit_bounding_box_zyx_0based')
        if not isinstance(ids,list) or len(ids)!=2 or len(set(i for i in ids if type(i) is int))!=2:
            continue
        segments=[session.segment(i) for i in ids]
        if not all(s is not None and s.recognition.get('method')=='source_ct_review_and_local_seeded_split'
                   and s.recognition.get('original_label_id')==report.get('original_label_id') for s in segments):
            continue
        if not (isinstance(box,list) and len(box)==3 and all(
             isinstance(b,list) and len(b)==2 and all(type(i) is int for i in b) and 0<=b[0]<b[1]<=n
             for b,n in zip(box,session.labels.shape))):
            continue
        # The 3D review box is bounded independently of the untrusted input size.
        if np.prod([b[1]-b[0] for b in box],dtype=np.int64)>1_000_000:
            continue
        overlap=[(max(b[0]-margin,r.start),min(b[1]+margin,r.stop)) for b,r in zip(box,roi)]
        if any(x>=y for x,y in overlap):
            continue
        # Retain a one-voxel halo around the recorded pair on either side of the
        # reviewed area, even when the user deliberately crops the current ROI.
        halo=tuple(slice(max(0,x-1),min(n,y+1)) for (x,y),n in zip(overlap,session.labels.shape))
        labels=session.labels[halo]
        if not all(np.any(labels==i) for i in ids):
            continue
        inner=tuple(slice(x-h.start,y-h.start) for (x,y),h in zip(overlap,halo))
        local=tuple(slice(x-r.start,y-r.start) for (x,y),r in zip(overlap,roi))
        yield ids,halo,inner,local


def protected_gap_mask(session, roi=None):
    roi=session.roi if roi is None else roi
    allowed=None
    for ids,halo,inner,local in _reviewed_gaps(session,roi,margin=0):
        labels=session.labels[halo]
        # A zero voxel near *both* reviewed bones is part of the guarded gap.
        # Avoid locking unrelated voids/skin in the same bounding box.
        k=ndi.generate_binary_structure(3,3)
        mask=(labels[inner]==0)
        for ident in ids:
            mask &= ndi.binary_dilation(labels==ident,structure=k)[inner]
        if not np.any(mask):
            continue
        if allowed is None:
            allowed=np.zeros(tuple(r.stop-r.start for r in roi),bool)
        allowed[local] |= mask
    return allowed


def trim_new_contacts(session, candidate, *, baseline=None):
    """Keep the recorded pair disjoint after two separate expansions.

    A pair of proposals can touch where neither of the new voxels originally
    had a neighbour from both bones. Reject newly proposed contact voxels in
    the recorded seam and its 3-voxel halo, retaining original contours.
    """
    total=0
    roi=session.roi
    if baseline is None:baseline=session.labels[roi]
    k=ndi.generate_binary_structure(3,3)
    for ids,halo,inner,local in _reviewed_gaps(session,roi,margin=3):
        halo_label=session.labels[halo].copy()
        overlap=[];sub=[]
        for h,r in zip(halo,roi):
            x=max(h.start,r.start);y=min(h.stop,r.stop)
            overlap.append(slice(x-h.start,y-h.start));sub.append(slice(x-r.start,y-r.start))
        halo_label[tuple(overlap)]=candidate[tuple(sub)]
        new=candidate[local]
        old=baseline[local]
        removed=np.zeros(new.shape,bool)
        for a,b in ((ids[0],ids[1]),(ids[1],ids[0])):
            removed |= (new==a)&(old==0)&ndi.binary_dilation(halo_label==b,structure=k)[inner]
        total+=int(np.count_nonzero(removed))
        new[removed]=0
    return total
