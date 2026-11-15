"""Local pretrained LiteMedSAM boundary proposals; no upload or remote inference.

Initial label masks provide prompts. Per-slice geometric checks reject large
deviations; those are quality screens, not calibrated measures of accuracy.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import time

import numpy as np
from scipy import ndimage as ndi
from .segmentation import fill_small_2d_holes, fill_bone_cavities

CHECKPOINT_SHA256='79d8c9dca6db4d69d3f905579e5250af05e859fff9c1f543e89a513c3028ce76'
MAX_MODEL_GROWTH_MM = 1.0
CT_BOUNDARY_MARGIN_MM = .8
CT_CORTEX_SEED_HU = 200
CT_CORTEX_CONNECT_HU = 150
CT_REFERENCE_CONTEXT_MM = 15


def ct_bone_reference(prior, image_hu, spacing, foreign=None):
    """Recover a dense cortical envelope only from components touching this bone."""
    # Seed on clear cortex, then follow connected weaker cortex. A single hard
    # threshold can open a thin cortical wall and erase the marrow behind it.
    dense = image_hu >= CT_CORTEX_CONNECT_HU
    if foreign is not None:dense &= ~foreign
    components, _ = ndi.label(dense, structure=ndi.generate_binary_structure(dense.ndim, dense.ndim))
    overlap = np.bincount(components[prior & (image_hu >= CT_CORTEX_SEED_HU) & dense], minlength=int(components.max())+1)
    overlap[0] = 0
    kept = np.flatnonzero(overlap > 0)
    supported = np.isin(components, kept) if len(kept) else np.zeros_like(prior)
    if prior.ndim == 3:
        return fill_bone_cavities(supported, spacing, None if foreign is None else ~foreign)
    return fill_small_2d_holes(supported, spacing)


def checkpoint_path():
    return Path(__file__).resolve().parents[1]/'models/lite_medsam.pth'


def model_ready():
    return checkpoint_path().is_file() and importlib.util.find_spec('torch') is not None and importlib.util.find_spec('timm') is not None


def torch_device(requested='auto'):
    import torch
    if requested=='auto':
        requested='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if requested=='cuda' and not torch.cuda.is_available():raise ValueError('当前 PyTorch 没有可用 CUDA 设备。')
    if requested=='mps' and not torch.backends.mps.is_available():raise ValueError('当前 PyTorch 没有可用 Apple MPS 设备。')
    return requested


@lru_cache(maxsize=1)
def load_model(requested='auto'):
    if not model_ready():raise ValueError('本地医学模型尚未安装。请运行“安装 AI”脚本后重启软件。')
    import torch
    import torch.nn as nn
    from .vendor.medsam_lite.tiny_vit_sam import TinyViT
    from .vendor.medsam_lite.mask_decoder import MaskDecoder
    from .vendor.medsam_lite.prompt_encoder import PromptEncoder
    from .vendor.medsam_lite.transformer import TwoWayTransformer
    path=checkpoint_path()
    if hashlib.sha256(path.read_bytes()).hexdigest()!=CHECKPOINT_SHA256:
        raise ValueError('模型权重校验失败，请重新下载官方权重。')
    device=torch_device(requested)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    if device=='mps':torch.mps.set_per_process_memory_fraction(0.35)
    if device=='cuda':
        torch.cuda.set_per_process_memory_fraction(0.5)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    model=nn.Module()
    model.image_encoder=TinyViT(img_size=256,in_chans=3,embed_dims=[64,128,160,320],depths=[2,2,6,2],
        num_heads=[2,4,5,10],window_sizes=[7,7,14,7],mlp_ratio=4.,drop_rate=0.,drop_path_rate=0.,
        use_checkpoint=False,mbconv_expand_ratio=4.0,local_conv_size=3,layer_lr_decay=0.8)
    model.prompt_encoder=PromptEncoder(embed_dim=256,image_embedding_size=(64,64),input_image_size=(256,256),mask_in_chans=16)
    model.mask_decoder=MaskDecoder(num_multimask_outputs=3,
        transformer=TwoWayTransformer(depth=2,embedding_dim=256,mlp_dim=2048,num_heads=8),
        transformer_dim=256,iou_head_depth=3,iou_head_hidden_dim=256)
    # Only tensor state dictionaries from the pinned public checkpoint are accepted.
    state=torch.load(path,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    model.to(device).eval()
    return model,device


def infer_slice(image_hu,box_xyxy,device='auto',window=(-500,2000)):
    import torch
    import torch.nn.functional as F
    model,device=load_model(device)
    height,width=image_hu.shape
    scale=256/max(height,width)
    nh,nw=int(height*scale+0.5),int(width*scale+0.5)
    gray=np.clip((image_hu.astype(np.float32)-window[0])/(window[1]-window[0]),0,1)
    tensor=torch.from_numpy(np.ascontiguousarray(gray))[None,None].to(device)
    with torch.inference_mode():
        tensor=F.interpolate(tensor,size=(nh,nw),mode='bilinear',align_corners=False)
        # Match upstream per-image min/max normalization before padding.
        tensor=(tensor-tensor.amin())/(tensor.amax()-tensor.amin()).clamp(min=1e-8)
        tensor=F.pad(tensor,(0,256-nw,0,256-nh)).repeat(1,3,1,1)
        embedding=model.image_encoder(tensor)
        box=np.asarray(box_xyxy,dtype=np.float32)*np.asarray([nw/width,nh/height,nw/width,nh/height])
        box_tensor=torch.as_tensor(box,dtype=torch.float32,device=device)[None,None,:]
        sparse,dense=model.prompt_encoder(points=None,boxes=box_tensor,masks=None)
        logits,scores=model.mask_decoder(image_embeddings=embedding,image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,multimask_output=False)
        logits=logits[...,:nh,:nw]
        probabilities=F.interpolate(logits,size=(height,width),mode='bilinear',align_corners=False).sigmoid()[0,0]
        result=probabilities.cpu().numpy()
        score=float(scores[0,0].item())
    return result,score,device


def screen_proposal(prior,probability,image_hu,spacing_yx,foreign=None,*,ct_reference=None,preserve_solid=True):
    """Keep CT-supported cortex and reject loss of sizeable separate bone islands."""
    reasons=[]
    if not np.isfinite(probability).all():
        return prior.copy(),{'accepted':False,'flags':['模型输出存在无效数值'],'area_ratio_to_initial':1.,'dice_against_initial_not_ground_truth':1.}
    prediction=probability>=0.5
    components,_=ndi.label(prediction)
    overlap=np.bincount(components[prior].ravel(),minlength=int(components.max())+1);overlap[0]=0
    keep=np.flatnonzero(overlap>0)
    raw_prediction=np.isin(components,keep) if len(keep) else np.zeros_like(prior)
    raw_area_ratio=float(raw_prediction.sum()/max(1,int(prior.sum())))
    prediction=fill_small_2d_holes(raw_prediction, spacing_yx) if preserve_solid else raw_prediction.copy()
    prior_envelope=fill_small_2d_holes(prior, spacing_yx) if preserve_solid else prior
    distance_outside=ndi.distance_transform_edt(~prior_envelope,sampling=spacing_yx)
    prediction &= distance_outside<=MAX_MODEL_GROWTH_MM
    if preserve_solid:
        # Keep a solid medullary interior while the model refines the surface.
        # This prevents isolated rejected/contracted slices making internal slots.
        depth=ndi.distance_transform_edt(prior_envelope,sampling=spacing_yx)
        prediction |= prior_envelope & (depth > 1.0)
    if ct_reference is None:
        ct_reference=ct_bone_reference(prior,image_hu,spacing_yx,foreign)
    coverage=float(np.count_nonzero(ct_reference & prior)/max(1,int(prior.sum())))
    reference_reliable=bool(ct_reference.any() and coverage>=.55)
    if not reference_reliable:reasons.append('CT 皮质边界依据不足，模型未应用')
    before_constraint=int(prediction.sum())
    if reference_reliable:
        ct_distance=ndi.distance_transform_edt(~ct_reference,sampling=spacing_yx)
        prediction &= ct_distance<=CT_BOUNDARY_MARGIN_MM
    else:
        # Without a reliable cortical envelope, do not grow into soft tissue.
        prediction &= prior_envelope
    # The 256px model can undershoot bright thin cortex. Keep existing dense
    # CT-supported portions of this label, including those beyond its proposed
    # edge. The learned contour refines lower-density areas around that core.
    prediction |= prior & (image_hu>=450)
    if foreign is not None:prediction &= ~foreign
    area_ratio=float(prediction.sum()/max(1,int(prior.sum())))
    dice=float(2*np.count_nonzero(prediction&prior)/max(1,int(prediction.sum()+prior.sum())))
    if not 0.55<=area_ratio<=1.8 or raw_area_ratio>1.8:reasons.append('轮廓变化过大')
    if dice<0.62:reasons.append('与初分割差异较大')
    islands,n=ndi.label(prior)
    sizes=np.bincount(islands.ravel());covered=np.bincount(islands[raw_prediction].ravel(),minlength=n+1)
    for i in range(1,n+1):
        if n>1 and sizes[i]>=max(20,prior.sum()*0.05) and covered[i]/sizes[i]<0.4:
            reasons.append('可能漏掉独立骨片');break
    return prediction,{'accepted':not reasons,'flags':reasons,
                       'area_ratio_to_initial':area_ratio,'dice_against_initial_not_ground_truth':dice,
                       'raw_model_area_ratio':raw_area_ratio,'max_model_growth_mm':MAX_MODEL_GROWTH_MM,
                       'ct_boundary_margin_mm':CT_BOUNDARY_MARGIN_MM,'ct_reference_coverage':coverage,
                       'ct_edge_constraint_applied':reference_reliable,
                       'excluded_exterior_voxels':max(0,before_constraint-int(prediction.sum()))}


def refine_segment(session,ident,progress=lambda *_:None,device='auto',max_slices=None,*,protected_gap=None,original_owners=None):
    if session.volume.is_mr:
        from .mr_tools import refine_mr_many
        local,reports=refine_mr_many(session,[ident],progress,device)
        if not reports:raise ValueError('该结构为空或已标记缺失，未启动 MRI 模型。')
        return local,reports[0]
    if session.segment(ident) is None:raise ValueError('请先选择一个骨标签。')
    roi=session.roi
    local=session.labels[roi].copy();hu=session.volume.hu[roi]
    original_source=local==ident
    if not original_source.any():raise ValueError('当前骨标签为空，请先进行初分割。')
    spacing=session.volume.spacing_zyx
    progress(0,'准备骨内髓腔与 CT 皮质边界')
    preserve_solid=bool(session.parameters.get('fill',True))
    available=local==0
    if protected_gap is not None:available &= ~protected_gap
    if original_owners is not None:available &= (original_owners==0)|(original_owners==ident)
    source=fill_bone_cavities(original_source,spacing,available,lambda value,text:progress(int(value*.04),text)) if preserve_solid else original_source
    # Compute the physical reference in a tight bounding box, using all three
    # planes to retain marrow when the cortical ring is open on an axial slice.
    box=[]
    for axis in range(3):
        occupied=np.flatnonzero(source.any(axis=tuple(i for i in range(3) if i!=axis)))
        margin=int(np.ceil(CT_REFERENCE_CONTEXT_MM/spacing[axis]))
        box.append(slice(max(0,int(occupied[0])-margin),min(source.shape[axis],int(occupied[-1])+margin+1)))
    box=tuple(box)
    foreign=(local[box]!=0)&~original_source[box]
    if protected_gap is not None:foreign |= protected_gap[box]
    if original_owners is not None:
        owners=original_owners[box];foreign |= (owners!=0)&(owners!=ident)
    ct_reference=np.zeros_like(source)
    ct_reference[box]=ct_bone_reference(source[box],hu[box],spacing,foreign)
    z_positions=np.flatnonzero(source.any(axis=(1,2)))
    if max_slices is not None and len(z_positions)>max_slices:
        z_positions=z_positions[np.linspace(0,len(z_positions)-1,max_slices,dtype=int)]
    entries=[];started=time.perf_counter()
    progress(5,'载入本地医学分割模型')
    _,actual_device=load_model(device)
    margin_y=max(8,int(np.ceil(15/spacing[1])));margin_x=max(8,int(np.ceil(15/spacing[2])))
    for number,z in enumerate(z_positions):
        progress(5+int(number/len(z_positions)*90),f'模型精修 {number+1} / {len(z_positions)} 层 · {actual_device.upper()}')
        prior=source[z]
        ys,xs=np.where(prior)
        y0,y1=max(0,int(ys.min())-margin_y),min(prior.shape[0],int(ys.max())+margin_y+1)
        x0,x1=max(0,int(xs.min())-margin_x),min(prior.shape[1],int(xs.max())+margin_x+1)
        cropped=prior[y0:y1,x0:x1]
        ys,xs=np.where(cropped)
        # Boxes extend slightly beyond the current edge to allow cortical recovery.
        box=[max(0,int(xs.min())-3),max(0,int(ys.min())-3),min(cropped.shape[1],int(xs.max())+4),min(cropped.shape[0],int(ys.max())+4)]
        probability,score,_=infer_slice(hu[z,y0:y1,x0:x1],box,device)
        view=local[z,y0:y1,x0:x1]
        excluded=(view!=0)&(view!=ident)
        if protected_gap is not None:excluded |= protected_gap[z,y0:y1,x0:x1]
        if original_owners is not None:
            owners=original_owners[z,y0:y1,x0:x1]
            excluded |= (owners!=0)&(owners!=ident)
        prediction,quality=screen_proposal(cropped,probability,hu[z,y0:y1,x0:x1],spacing[1:],excluded,
            ct_reference=ct_reference[z,y0:y1,x0:x1],preserve_solid=preserve_solid)
        # Model score is recorded for inspection, never treated as a calibrated accuracy.
        accepted=quality['accepted']
        if accepted:
            editable=(view==0)|(view==ident)
            view[(view==ident)&~prediction]=0
            view[prediction&editable&~excluded]=ident
        entries.append({'slice_1based':int(z+roi[0].start+1),'model_score_uncalibrated':score,**quality})
    # Compare change ratios, rather than raw areas, so genuine anatomical taper
    # already present in CT is not confused with model-induced slice flicker.
    by_z={e['slice_1based']:e for e in entries}
    for entry in entries:
        z1=entry['slice_1based']
        neighbors=[by_z[k]['area_ratio_to_initial'] for k in [z1-1,z1+1] if k in by_z and by_z[k]['accepted']]
        if entry['accepted'] and len(neighbors)==2:
            relative=entry['area_ratio_to_initial']/max(1e-8,float(np.median(neighbors)))
            if not 0.7<=relative<=1.4:
                z=z1-roi[0].start-1
                view=local[z];view[view==ident]=0;view[original_source[z]]=ident
                entry['accepted']=False;entry['flags'].append('层间轮廓变化异常')
    report={'model':'LiteMedSAM','checkpoint_sha256':CHECKPOINT_SHA256,'device':actual_device,
            'precision':'float32; CUDA TF32 disabled',
            'preserved_prior_density_threshold_hu':450,
            'boundary_method':'ct_bounded_v3','max_model_growth_mm':MAX_MODEL_GROWTH_MM,
            'ct_boundary_margin_mm':CT_BOUNDARY_MARGIN_MM,'preserve_solid_interior':preserve_solid,
            'ct_cortex_seed_hu':CT_CORTEX_SEED_HU,'ct_cortex_connect_hu':CT_CORTEX_CONNECT_HU,
            'ct_reference_context_mm':CT_REFERENCE_CONTEXT_MM,
            'label_id':ident,'label_name':session.segment(ident).name,'slices':entries,
            'accepted_slices':sum(e['accepted'] for e in entries),'flagged_slices':sum(not e['accepted'] for e in entries),
            'seconds':time.perf_counter()-started,
            'limitations':'No ground-truth comparison; acceptance checks are consistency screens, not proof of improved anatomical accuracy.'}
    progress(100,'模型精修完成，异常层保留原分割')
    return local,report


def refine_many(session,ids,progress=lambda *_:None,device='auto'):
    """Build a batch candidate in a cropped working copy; cancellation is atomic."""
    if session.volume.is_mr:
        from .mr_tools import refine_mr_many
        return refine_mr_many(session,ids,progress,device)
    from dataclasses import replace
    from .model import Session,Volume
    from .gap_guard import protected_gap_mask, trim_new_contacts
    if not ids:raise ValueError('勾选的骨标签为空。')
    roi=session.roi
    original_owners=session.labels[roi]
    affine=session.volume.affine_lps.copy()
    affine[:3,3]=session.volume.world([s.start for s in roi])
    v=Volume(session.volume.hu[roi],affine,session.volume.description)
    work=Session(v,labels=session.labels[roi].copy(),segments=[replace(s) for s in session.segments],
                 joint=session.joint,side=session.side,parameters=dict(session.parameters))
    reports=[]
    protected=protected_gap_mask(session)
    recover_edges=session.parameters.get('recover_edges',True)
    model_budget=85 if recover_edges else 100
    for i,ident in enumerate(ids):
        local,report=refine_segment(work,ident,
            lambda amount,text:progress(int((i+amount/100)/len(ids)*model_budget),f'{i+1}/{len(ids)} 骨 · {text}'),device,
            protected_gap=protected,original_owners=original_owners)
        work.labels[:]=local
        for entry in report['slices']:entry['slice_1based']+=roi[0].start
        reports.append(report)
    trimmed=trim_new_contacts(session,work.labels)
    if trimmed:reports[0]['reviewed_gap_contact_proposals_rejected']=trimmed
    if recover_edges:
        from .recovery import recover_ct_edges
        completed,recovery=recover_ct_edges(session,ids,
            lambda amount,text:progress(85+int(amount*.15),text),labels=work.labels,original_owners=original_owners)
        by_id={row['id']:row for row in recovery['bones']}
        for report in reports:
            report['ct_edge_recovery']=by_id.get(report['label_id'],{})
            report['ct_recovery_method']=recovery['method']
            recovered_by_z=np.count_nonzero((completed==report['label_id'])&(work.labels==0),axis=(1,2))
            for entry in report['slices']:
                entry['ct_recovered_voxels']=int(recovered_by_z[entry['slice_1based']-roi[0].start-1])
        work.labels[:]=completed
        recovery['input_stage']='model_output'
        reports[0]['batch_ct_recovery']=recovery
    return work.labels,reports
