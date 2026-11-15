"""Bounded native-resolution whole-leg initial segmentation.

Threshold/outline cleanup and axial filling are slab-local. Component IDs are
then reconciled using exactly the 26-neighbour pairs across the slab boundaries;
the volume filter is applied only after those counts have been merged.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import ndimage as ndi

from .preprocessing import patient_region, exclude_external_attachments
from .resources import detect_memory, memory_policy
from .segmentation import fill_small_2d_holes


CONNECTIVITY = ndi.generate_binary_structure(3, 3)
SLAB_REPORTED_CAP = 20_000_000
MAX_COMPONENT_LABELS = 128
PERIPHERAL_DEPTH_MM = 5.0
RAIL_MAX_INTERIOR_FRACTION = .4


def whole_leg_plan(roi, policy=None):
    """Budget one uint32 scratch volume, plus a bounded array of slab temporaries."""
    if policy is None:
        policy=memory_policy(detect_memory())
    z,y,x=(r.stop-r.start for r in roi)
    n=z*y*x
    ops=policy['operations']['initial']
    available=policy['memory_available_bytes']
    if available is None:
        available=policy['memory_total_bytes']//2
    reserve=policy['reserved_bytes']+policy['fixed_overhead_bytes']
    # Stitching and output conversion are sequential (two and four bytes per
    # voxel); session commit can transiently hold copies for undo compression.
    retained_budget=6*n
    working=max(0,available-reserve-retained_budget)
    cap=min(SLAB_REPORTED_CAP,ops['limit_voxels'],working//40)
    slab=min(z,80,cap//(x*y))
    if slab<min(z,2):
        raise ValueError('全腿处理的内存不足。请释放内存，或缩小横向范围后重试（可保留全部层）。')
    return {'z':z,'plane_voxels':x*y,'full_voxels':n,'slab_depth':slab,'max_slab_voxels':slab*x*y,
            'estimated_retained_bytes':retained_budget,'available_bytes':available}


class _LabelUnion:
    def __init__(self):
        self.parent=[0]
        self.rank=[0]
        self.counts=[0]
        self.anchor=[0]
        self.uncertain=[0]
        self.lower=[(0,0,0)]
        self.upper=[(0,0,0)]

    def add(self, counts, anchor, uncertain, boxes, z0):
        start=len(self.parent)-1
        self.parent.extend(range(start+1,start+1+len(counts)))
        self.rank.extend([0]*len(counts))
        self.counts.extend(int(c) for c in counts)
        self.anchor.extend(int(c) for c in anchor)
        self.uncertain.extend(int(c) for c in uncertain)
        self.lower.extend((b[0].start+z0,b[1].start,b[2].start) for b in boxes)
        self.upper.extend((b[0].stop+z0,b[1].stop,b[2].stop) for b in boxes)
        return start

    def root(self,a):
        while self.parent[a]!=a:
            self.parent[a]=self.parent[self.parent[a]]
            a=self.parent[a]
        return a

    def connect(self,a,b):
        a,b=self.root(a),self.root(b)
        if a==b:return
        if self.rank[a]<self.rank[b]:a,b=b,a
        self.parent[b]=a
        if self.rank[a]==self.rank[b]:self.rank[a]+=1


def _cut_pairs(before, after):
    h,w=before.shape
    pairs=[]
    for y in (-1,0,1):
        for x in (-1,0,1):
            a=before[max(0,-y):min(h,h-y),max(0,-x):min(w,w-x)]
            b=after[max(0,y):min(h,h+y),max(0,x):min(w,w+x)]
            found=(a!=0)&(b!=0)
            if found.any():pairs.append(np.column_stack((a[found],b[found])))
    return np.unique(np.concatenate(pairs),axis=0) if pairs else ()


def peripheral_rail_candidates(volume, eligible, totals, anchors, uncertain, lower, upper):
    """Find shallow outer tracks supported by a long thin rail in the same band."""
    spacing=volume.spacing_zyx
    axis_xyz=volume.affine_lps[:3,:3]/spacing[::-1]
    lr_xyz=int(np.argmax(np.abs(axis_xyz[0,:2])))
    ap_xyz=1-lr_xyz
    # Component-aligned rectangles are insufficient evidence on an oblique grid.
    if abs(axis_xyz[0,lr_xyz])<.95 or abs(axis_xyz[1,ap_xyz])<.95:
        return np.zeros_like(eligible)
    lr,ap=2-lr_xyz,2-ap_xyz
    supported=eligible&(anchors>=totals*.55)&(uncertain==0)
    if np.count_nonzero(supported)<2:
        return np.zeros_like(eligible)
    lo=float(lower[supported,lr].min());hi=float(upper[supported,lr].max())
    right=upper[:,lr]<=lo-6/spacing[lr]
    left=lower[:,lr]>=hi+6/spacing[lr]
    spans=(upper-lower)*spacing
    shallow=eligible&(anchors<totals*RAIL_MAX_INTERIOR_FRACTION)&(uncertain==0)
    slender=shallow&(spans[:,ap]<=22)
    tracks=slender & (right|left) & (spans[:,0]>=150)&(spans[:,0]>=3*np.maximum(spans[:,lr],spans[:,ap]))
    marked=np.zeros_like(eligible)
    for index in np.flatnonzero(tracks):
        nearby=(right if right[index] else left).copy()
        for axis in (lr,ap):
            separation=np.maximum(lower[index,axis]-upper[:,axis],lower[:,axis]-upper[index,axis])
            nearby &= separation*spacing[axis]<=6
        marked|=slender&nearby
    return marked


def prepare_whole_leg(session, parameters, progress=lambda *_: None, *, slab_depth=None):
    volume=session.volume
    if volume.is_mr:raise ValueError('全腿 CT 阈值流程不能用于 MRI。')
    roi=session.roi
    depth,y,x=(b.stop-b.start for b in roi)
    if slab_depth is None:
        slab_depth=whole_leg_plan(roi)['slab_depth']
    threshold,upper,minimum=[parameters[k] for k in ('threshold','upper','min_mm3')]
    if not np.isfinite([threshold,upper,minimum]).all() or threshold>=upper or minimum<0:
        raise ValueError('阈值上下限或最小体积无效。')
    if not isinstance(slab_depth,int) or slab_depth<1 or slab_depth*y*x>SLAB_REPORTED_CAP:
        raise ValueError('全腿分段设置超出工作预算。')
    chunks=math.ceil(depth/slab_depth)
    staging=np.empty((depth,y,x),np.uint32)
    uf=_LabelUnion()
    shared=0
    prep={'method':'whole_leg_slab_cleanup_v1','roi_zyx':[[b.start,b.stop] for b in roi],
          'removed_labeled_voxels':0,'removed_external_components':0,'ambiguous_attachment_components':0,
          'preserved_uncertain_slices_1based':[],'supported_slices':0,'outline_fallback_slabs':[],
          'unanchored_peripheral_components_removed':0,'unanchored_peripheral_voxels_removed':0}
    last_progress=0
    for index,z0 in enumerate(range(0,depth,slab_depth)):
        z1=min(depth,z0+slab_depth)
        box=(slice(roi[0].start+z0,roi[0].start+z1),*roi[1:])
        def step(p,text):
            nonlocal last_progress
            last_progress=max(last_progress,min(90,int(90*(index+p/100)/chunks)))
            progress(last_progress,f'全腿 {index+1}/{chunks} 段 · {text}')
        values=volume.hu[box]
        keep=None
        if parameters.get('remove_table',True):
            try:
                keep,part=patient_region(volume,box,lambda value,text:step(value*.4,text))
            except ValueError as exc:
                if not str(exc).startswith('没有可靠的人体轮廓'):
                    raise
                # Some scans intentionally include empty acquisition slabs at
                # an end. Retain raw HU; with no trustworthy outline there is
                # no safe way to select a patient-bound component in this slab.
                prep['outline_fallback_slabs'].append([box[0].start+1,box[0].stop])
                prep['preserved_uncertain_slices_1based'].extend(range(box[0].start+1,box[0].stop+1))
        mask=(values>=threshold)&(values<=upper)
        if keep is not None:
            mask,attached=exclude_external_attachments(mask,keep,volume.spacing_zyx,lambda p,t:step(40+p*.28,t))
            # The conservative outline includes a small peripheral margin. A
            # rail may occupy that margin for the full length of an acquisition
            # without a deep patient anchor, even if its exterior portion was
            # just removed. Filter only after all slabs have been joined, so a
            # real bone with a deep anchor elsewhere in the scan is retained.
            prep['supported_slices']+=part['supported_slices']
            prep['preserved_uncertain_slices_1based'].extend(part['preserved_uncertain_slices_1based'])
            for key in ('removed_labeled_voxels','removed_external_components','ambiguous_attachment_components'):
                prep[key]+=attached[key]
        if parameters.get('fill',True):
            for z,plane in enumerate(mask):
                if z%16==0:step(69+int(11*z/len(mask)),'填充骨内部')
                mask[z]=fill_small_2d_holes(plane,volume.spacing_zyx[1:])
        if keep is not None:mask&=keep
        step(82,'标记本段连通骨块')
        cc,n=ndi.label(mask,structure=CONNECTIVITY)
        sizes=np.zeros(n+1,np.int64);anchors=np.zeros_like(sizes);unsure=np.zeros_like(sizes)
        uncertain=(set(part['preserved_uncertain_slices_1based']) if keep is not None else set())
        for i,plane in enumerate(cc):
            if i%8==0:step(82+int(8*i/len(cc)),'累加各骨块与人体轮廓的位置')
            histogram=np.bincount(plane.ravel(),minlength=n+1)
            sizes+=histogram
            if not histogram[1:].any():continue
            if keep is None or box[0].start+i+1 in uncertain:
                unsure+=histogram
            else:
                depth_mm=ndi.distance_transform_edt(np.pad(keep[i],1),sampling=volume.spacing_zyx[1:])[1:-1,1:-1]
                anchors+=np.bincount(plane[depth_mm>PERIPHERAL_DEPTH_MM],minlength=n+1)
        offset=uf.add(sizes[1:],anchors[1:],unsure[1:],ndi.find_objects(cc),z0)
        lookup=np.arange(offset,offset+n+1,dtype=np.uint32);lookup[0]=0
        for i,plane in enumerate(cc):staging[z0+i]=lookup[plane]
        if z0:
            matches=_cut_pairs(staging[z0-1],staging[z0])
            shared+=len(matches)
            for a,b in matches:uf.connect(int(a),int(b))
        del mask,cc,keep
        step(100,'衔接相邻采集段')
    progress(91,'按全腿连续性合并并筛选骨块')
    roots=np.fromiter((uf.root(i) for i in range(len(uf.parent))),count=len(uf.parent),dtype=np.int32)
    totals=np.zeros(len(roots),np.int64)
    np.add.at(totals,roots,np.asarray(uf.counts,np.int64))
    eligible=(totals>0)&(totals*volume.voxel_mm3>=minimum)
    if parameters.get('remove_table',True):
        anchor=np.zeros(len(roots),np.int64);unknown=np.zeros_like(anchor)
        np.add.at(anchor,roots,np.asarray(uf.anchor,np.int64));np.add.at(unknown,roots,np.asarray(uf.uncertain,np.int64))
        # Join all support across slab boundaries before deciding which
        # components have no interior anchor. Also recognize repeated, long
        # thin tracks outside the range of all strongly anchored bone labels.
        lower=np.full((len(roots),3),max(staging.shape),np.int32)
        upper=np.zeros_like(lower)
        np.minimum.at(lower,roots,np.asarray(uf.lower,np.int32))
        np.maximum.at(upper,roots,np.asarray(uf.upper,np.int32))
        unsupported=eligible&(anchor==0)&(unknown==0)
        rails=peripheral_rail_candidates(volume,eligible,totals,anchor,unknown,lower,upper)
        prep['outer_rail_candidates_removed']=int(np.count_nonzero(rails&~unsupported))
        unsupported|=rails
        prep['unanchored_peripheral_components_removed']=int(np.count_nonzero(unsupported))
        prep['unanchored_peripheral_voxels_removed']=int(totals[unsupported].sum())
        prep['removed_labeled_voxels']+=prep['unanchored_peripheral_voxels_removed']
        eligible &= ~unsupported
    keep_ids=np.flatnonzero(eligible)
    keep_ids=keep_ids[np.argsort(-totals[keep_ids],kind='stable')]
    if not len(keep_ids):raise ValueError('全腿范围没有找到符合体积阈值的骨块。')
    if len(keep_ids)>MAX_COMPONENT_LABELS:
        raise ValueError(f'全腿范围有 {len(keep_ids)} 个候选碎块；请提高最小骨块体积或缩小范围。')
    siblings=np.bincount(roots,minlength=len(roots))
    lut=np.zeros(len(roots),np.uint16);lut[keep_ids]=np.arange(1,len(keep_ids)+1,dtype=np.uint16)
    lut=lut[roots]
    stitched=np.empty(staging.shape,np.uint16)
    for i,plane in enumerate(staging):
        if i%64==0:progress(92+int(7*i/depth),'写入全腿统一标签')
        stitched[i]=lut[plane]
    prep.update(slab_depth=slab_depth,slabs=chunks,paired_cross_slab_component_ids=shared,
                provisional_components=len(roots)-1,global_component_count=len(keep_ids),
                labels_crossing_slab_boundaries=[i for i,r in enumerate(keep_ids,1)
                      if siblings[r]>1],
                uses_original_resolution=True,
                skipped_automatic_recovery_and_knee_naming=True,
                limitation='Whole-leg components are editable, not standardized anatomical names; heel, toe, pelvic and touching bones require CT review.')
    progress(100,'全腿连通骨块已生成，请按原片复核和命名')
    return {'segmentation':(stitched,len(keep_ids)),'recognition':[],
            'preprocessing':prep,'repair':None}
