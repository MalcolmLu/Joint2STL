"""Local native-sagittal MRI correction from positive/negative prompts.

Outputs are private candidates, never edits to a Session. This is an interactive
model, not a trained bone-name/ACL/PCL classifier or an accuracy certificate.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
from numbers import Integral
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from . import ai
from .mr_policy import require_mri_modeling_source,modeling_source_metadata

CHECKPOINT_BYTES=15_977_486
CHECKPOINT_SHA256='43f57ee8fa8ec529c31be281e06749f9e629b30157bbbcc9baf200cddec1acbe'
CHECKPOINT_URL=('https://huggingface.co/spaces/halleewong/ScribblePrompt/resolve/'
                '706d6559618e5f1104565a6a19c9ea069fca7454/checkpoints/'
                'ScribblePrompt_unet_v1_nf192_res128.pt')


def checkpoint_path():
    return Path(__file__).resolve().parents[1]/'models/scribbleprompt_unet.pth'


def correction_model_ready(*,verify=False):
    path=checkpoint_path()
    try:
        return (path.is_file() and path.stat().st_size==CHECKPOINT_BYTES
                and importlib.util.find_spec('torch') is not None
                and (not verify or hashlib.sha256(path.read_bytes()).hexdigest()==CHECKPOINT_SHA256))
    except OSError:
        return False


@lru_cache(maxsize=2)
def _load_model(device):
    import torch
    from .vendor.scribbleprompt.network import UNet
    if not correction_model_ready(verify=True):
        raise ValueError('局部纠错模型未安装或校验失败，请运行 scripts/setup_correction_model.py。')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    if device=='mps':torch.mps.set_per_process_memory_fraction(.35)
    elif device=='cuda':
        torch.cuda.set_per_process_memory_fraction(.5)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    state=torch.load(checkpoint_path(),map_location='cpu',weights_only=True)
    if not isinstance(state,dict) or not all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items()):
        raise ValueError('局部纠错权重必须为纯张量字典。')
    model=UNet(in_channels=5,out_channels=1,features=[192,192,192,192])
    model.load_state_dict(state,strict=True)
    return model.to(device).eval()


def _xy_points(points,shape):
    result=[]
    for point in points:
        if len(point)!=2 or any(not isinstance(v,Integral) or isinstance(v,(bool,np.bool_)) for v in point):
            raise ValueError('MRI 提示点必须为整数 XY 坐标。')
        x,y=map(int,point)
        if not 0<=x<shape[1] or not 0<=y<shape[0]:raise ValueError('MRI 提示点超出原片。')
        result.append((x,y))
    return result


def _prepare(raw,box,positive_xy,negative_xy):
    """Map a local native crop to the model grid, retaining the inverse map."""
    import torch
    from torch.nn import functional as F
    raw=np.asarray(raw)
    if raw.ndim!=2 or raw.dtype.kind not in 'uif':raise ValueError('MRI 纠错需要二维数值原片。')
    if len(box)!=4 or any(not isinstance(v,Integral) or isinstance(v,(bool,np.bool_)) for v in box):
        raise ValueError('MRI 纠错范围必须为整数 XYXY 坐标。')
    x0,y0,x1,y1=map(int,box)
    if not (0<=x0<x1<=raw.shape[1] and 0<=y0<y1<=raw.shape[0]):raise ValueError('MRI 纠错范围超出原片。')
    positive_xy=_xy_points(positive_xy,raw.shape);negative_xy=_xy_points(negative_xy,raw.shape)
    if set(positive_xy)&set(negative_xy):raise ValueError('同一位置不能同时标为保留和排除。')
    pad=max(10,int(.15*max(x1-x0,y1-y0)))
    cx0,cy0=max(0,x0-pad),max(0,y0-pad)
    cx1,cy1=min(raw.shape[1],x1+pad),min(raw.shape[0],y1+pad)
    crop=np.asarray(raw[cy0:cy1,cx0:cx1],np.float64)
    finite=np.isfinite(crop)
    if not finite.any():raise ValueError('MRI 纠错范围内没有有效信号。')
    low,high=(float(v) for v in np.percentile(crop[finite],[1,99]))
    spread=high-low
    if not np.isfinite(spread):raise ValueError('MRI 信号范围无法归一化。')
    image=np.clip((np.where(finite,crop,low)-low)/(spread if spread>0 else 1.),0,1).astype(np.float32)
    channels=torch.zeros((1,5,128,128),dtype=torch.float32)
    channels[:,:1]=F.interpolate(torch.from_numpy(image)[None,None],(128,128),mode='bilinear',align_corners=False)
    sx,sy=128/crop.shape[1],128/crop.shape[0]
    bx0,by0,bx1,by1=(int(v) for v in ((x0-cx0)*sx,(y0-cy0)*sy,(x1-cx0)*sx,(y1-cy0)*sy))
    channels[0,1,by0:by1,bx0:bx1]=1
    for channel,points in ((2,positive_xy),(3,negative_xy)):
        for x,y in points:
            ix,iy=int((x-cx0)*sx),int((y-cy0)*sy)
            if not (0<=ix<128 and 0<=iy<128):raise ValueError('提示点超出当前局部纠错范围。')
            channels[0,channel,iy,ix]=1
    if bool((channels[0,2]*channels[0,3]).any()):
        raise ValueError('正负提示在模型网格上过近，请稍微分开或缩小纠错范围。')
    return channels,{
        'crop_xyxy_0based':[cx0,cy0,cx1,cy1],'box_xyxy_0based':[x0,y0,x1,y1],
        'model_size_yx':[128,128],'normalization_window':[low,high],
        'positive_xy_0based':[list(p) for p in positive_xy],
        'negative_xy_0based':[list(p) for p in negative_xy],
        'nonfinite_replaced':int((~finite).sum()),'degenerate_signal':high<=low,
    }


def infer_correction(raw,box,positive_xy,negative_xy=(),device='auto'):
    """Infer from one acquired slice; resize only model input/output, not MRI."""
    if not correction_model_ready():raise ValueError('MRI 局部纠错环境未就绪，请先运行 Install_AI_Mac.command 或 Install_NVIDIA_Windows.cmd 安装 AI 环境。')
    import torch
    from torch.nn import functional as F
    if device not in ('auto','cpu','mps','cuda'):raise ValueError('无效 MRI 模型计算设备。')
    channels,detail=_prepare(raw,box,positive_xy,negative_xy)
    result=np.zeros(np.asarray(raw).shape,np.float32)
    if detail['degenerate_signal']:
        return result,{**detail,'device':'not_run','inference_performed':False}
    requested=ai.torch_device(device)
    def run(actual):
        with torch.inference_mode():
            logits=_load_model(actual)(channels.to(actual))
            cx0,cy0,cx1,cy1=detail['crop_xyxy_0based']
            prob=F.interpolate(logits,size=(cy1-cy0,cx1-cx0),mode='bilinear',align_corners=False).sigmoid()[0,0]
            if not bool(torch.isfinite(prob).all()):raise ValueError('局部纠错模型返回无效概率。')
            return prob.cpu().numpy()
    actual=requested
    try:prob=run(actual)
    except (RuntimeError,NotImplementedError) as exc:
        if actual!='mps' or not any(word in str(exc).lower() for word in ('mps','metal','mpsgraph')):raise
        detail['mps_fallback_reason']=str(exc)[:1500]
        _load_model.cache_clear()
        try:torch.mps.empty_cache()
        except RuntimeError:pass
        actual='cpu';prob=run(actual)
    cx0,cy0,cx1,cy1=detail['crop_xyxy_0based'];result[cy0:cy1,cx0:cx1]=prob
    return result,{**detail,'device':actual,'inference_performed':True}


def _points_zyx(points,session,index):
    result=[]
    for point in points:
        if len(point)!=3 or any(not isinstance(v,Integral) or isinstance(v,(bool,np.bool_)) for v in point):
            raise ValueError('MRI 提示点必须为整数 ZYX 坐标。')
        point=tuple(map(int,point))
        if point[0]!=index or any(not r.start<=v<r.stop for v,r in zip(point,session.roi)):
            raise ValueError('提示点必须位于当前原始矢状采集层和范围内。')
        result.append((point[2],point[1]))
    return result


def propose_correction(session,ident,index,positive_zyx=(),negative_zyx=(),*,box=None,device='auto',automatic=False,_layer_only=False):
    """Return a private ROI proposal. Foreign ownership and absent labels win.

    Automatic refinement caps outward growth and rejects large changes. Explicit
    correction points can request a larger local repair, recorded as unreviewed.
    Enclosed bone holes are filled only within this acquired sagittal contour.
    """
    if not session.volume.is_mr:raise ValueError('局部 MRI 纠错仅支持 MR。')
    require_mri_modeling_source(session,'axial')
    if not isinstance(ident,Integral) or isinstance(ident,(bool,np.bool_)) or not 1<=ident<=65535:
        raise ValueError('请选择有效 MRI 结构标签。')
    segment=session.segment(ident)
    if segment is None or segment.kind not in ('bone','ligament'):raise ValueError('请选择现有骨或韧带标签。')
    if segment.presence=='absent':raise ValueError('已标记缺失的结构不能生成标注。')
    if not isinstance(index,Integral) or isinstance(index,(bool,np.bool_)) or not session.roi[0].start<=index<session.roi[0].stop:
        raise ValueError('请选择当前范围内的原始矢状采集层。')
    index=int(index);raw=session.volume.values[index];owners=session.labels[index]
    positive=_points_zyx(positive_zyx,session,index);negative=_points_zyx(negative_zyx,session,index)
    if any(owners[y,x] not in (0,ident) for x,y in positive):raise ValueError('保留点属于其他标签，请先核对或调整归属。')
    prior=owners==ident
    roi_mask=np.zeros(prior.shape,bool);roi_mask[session.roi[1:]]=True;prior &= roi_mask
    explicit_positive=bool(positive)
    if not positive:
        if not prior.any():raise ValueError('空标签需要至少一个保留点，或先用 M 框选。')
        distance=ndi.distance_transform_edt(prior,sampling=session.volume.spacing_zyx[1:])
        for x,y in negative:distance[y,x]=0
        y,x=np.unravel_index(distance.argmax(),distance.shape);positive=[(int(x),int(y))]
    if box is None:
        coords=np.argwhere(prior).tolist()+[[y,x] for x,y in positive+negative]
        coords=np.asarray(coords);padding=np.maximum(2,np.ceil(2/session.volume.spacing_zyx[1:]).astype(int))
        lo=np.maximum(coords.min(axis=0)-padding,[session.roi[1].start,session.roi[2].start])
        hi=np.minimum(coords.max(axis=0)+padding+1,[session.roi[1].stop,session.roi[2].stop])
        box=[int(lo[1]),int(lo[0]),int(hi[1]),int(hi[0])]
    x0,y0,x1,y1=box
    if not (session.roi[2].start<=x0<x1<=session.roi[2].stop and session.roi[1].start<=y0<y1<=session.roi[1].stop):
        raise ValueError('纠错范围必须包含在当前黄色范围内。')
    probability,detail=infer_correction(raw,box,positive,negative,device)
    probability=np.asarray(probability)
    if probability.shape!=raw.shape or not np.isfinite(probability).all():raise ValueError('局部纠错模型概率尺寸或数值无效。')
    scope=np.zeros(prior.shape,bool);scope[y0:y1,x0:x1]=True
    available=((owners==0)|(owners==ident)) & roi_mask
    prediction=(probability>=.5)&scope&available
    cc,n=ndi.label(prediction);keep={int(cc[y,x]) for x,y in positive if cc[y,x]}
    flags=[]
    if not keep:flags.append('foreground_prompt_not_supported')
    if automatic and segment.kind=='bone':
        overlap=np.bincount(cc[prior],minlength=n+1)*float(np.prod(session.volume.spacing_zyx[1:]))
        supported=np.flatnonzero(overlap>=5.)
        if any(k and k not in keep for k in supported):flags.append('unprompted_existing_fragment_needs_review')
    prediction=np.isin(cc,list(keep)) if keep else np.zeros(prior.shape,bool)
    if segment.kind=='bone':prediction=ndi.binary_fill_holes(prediction)&available&scope
    for x,y in negative:prediction[y,x]=False
    if keep and any(not prediction[y,x] for x,y in positive):flags.append('some_foreground_prompts_not_supported')
    if automatic:
        growth=2. if segment.kind=='bone' else 1.
        inside_contour=ndi.binary_fill_holes(prior) if segment.kind=='bone' else prior
        allowed=ndi.distance_transform_edt(~prior,sampling=session.volume.spacing_zyx[1:])<=growth
        prediction &= allowed|inside_contour
    original=prior&scope
    ratio=float(prediction.sum()/max(1,int(original.sum())))
    if automatic and not .5<=ratio<=2.:flags.append('large_model_change_needs_review')
    if not detail['inference_performed']:flags.append('no_valid_signal_for_model')
    accepted=not flags
    original_local=session.labels[index][session.roi[1:]]
    # Batch refinement allocates the 3D working volume once; each model call
    # needs only its native 2D plane, avoiding repeated full-volume copies.
    candidate=original_local.copy() if _layer_only else session.labels[session.roi].copy()
    current=candidate if _layer_only else candidate[index-session.roi[0].start]
    if accepted:
        local_scope=scope[session.roi[1:]];local_prediction=prediction[session.roi[1:]]
        current[(current==ident)&local_scope]=0;current[local_prediction]=ident
    entry={**detail,'fixed_axis':0,'native_acquisition_plane':True,'fixed_index_0based':index,'slice_1based':index+1,
           'accepted':accepted,'flags':flags,'area_ratio_to_prior_not_accuracy':ratio,'automatic':bool(automatic),
           'foreground_prompt_source':'explicit' if explicit_positive else 'existing-mask interior',
           'bone_fill_acquired_plane_only':segment.kind=='bone','reviewed':False}
    before=original_local if _layer_only else session.labels[session.roi]
    report={'method':'mr_local_positive_negative_correction_v1','model':'ScribblePrompt-UNet v1','modality':'MR',
            'checkpoint_sha256':CHECKPOINT_SHA256,'label_id':int(ident),'label_name':segment.name,'kind':segment.kind,'structure':segment.structure,
            'device':detail['device'],'fixed_axis':0,'plane':'axial','native_acquisition_plane':True,
            'slices':[entry],'accepted_slices':int(accepted),'flagged_slices':int(not accepted),'reviewed':False,
            'added_voxels':int(np.count_nonzero((candidate==ident)&(before!=ident))),
            'removed_voxels':int(np.count_nonzero((candidate!=ident)&(before==ident))),
            'modeling_source':modeling_source_metadata(session),
            'limitations':'Prompt-assisted candidate, not trained named anatomy or expert-reference accuracy; no diagnosis of presence.'}
    return candidate,report


def stored_points(session,ident=None,index=None):
    """Ignore malformed legacy metadata instead of breaking painting/rendering."""
    result=[]
    entries=session.parameters.get('mr_correction_points',[])
    if not isinstance(entries,list):return result
    for entry in entries:
        if not isinstance(entry,dict) or type(entry.get('positive')) is not bool:continue
        label=entry.get('label_id');point=entry.get('zyx')
        if type(label) is not int or session.segment(label) is None or not isinstance(point,(list,tuple)) or len(point)!=3:continue
        if session.segment(label).presence=='absent':continue
        if any(type(v) is not int or not 0<=v<n for v,n in zip(point,session.labels.shape)):continue
        if ident is not None and label!=ident:continue
        if index is not None and point[0]!=index:continue
        result.append({'label_id':label,'zyx':list(point),'positive':entry['positive']})
    return result


def refine_with_correction(session,ids,progress=lambda *_:None,device='auto'):
    """Conservative native-plane batch refinement with original ownership.

    Each target reads the same original labels, so a label cannot claim voxels
    another target gives up earlier in the batch. Large changes remain flagged.
    """
    if not session.volume.is_mr:raise ValueError('局部 MRI 精修仅支持 MR。')
    require_mri_modeling_source(session,'axial')
    ids=list(dict.fromkeys(ids))
    if not ids:raise ValueError('请选择需要精修的结构。')
    if any(session.segment(ident) is None for ident in ids):raise ValueError('MRI 精修目标标签不存在。')
    owners=session.labels[session.roi];candidate=owners.copy();reports=[]
    total=sum(int(np.any(owners==ident,axis=(1,2)).sum()) for ident in ids);done=0
    for ident in ids:
        segment=session.segment(ident);entries=[];report=None
        if segment.presence=='absent':candidate[candidate==ident]=0
        if segment.presence!='absent':
            for index in np.flatnonzero(np.any(owners==ident,axis=(1,2))):
                z=int(index+session.roi[0].start)
                progress(int(100*done/max(1,total)),f'MRI 局部精修 · {segment.name} · 原始层 {z+1}')
                local,one=propose_correction(session,ident,z,device=device,automatic=True,_layer_only=True)
                original=owners[index];target=candidate[index]
                prediction=local==ident
                target[(target==ident)&~prediction]=0
                target[prediction&((original==0)|(original==ident))]=ident
                entries.extend(one['slices']);report=one;done+=1
        if report is None:
            report={'method':'mr_local_positive_negative_correction_v1','model':'ScribblePrompt-UNet v1','modality':'MR',
                    'label_id':int(ident),'label_name':segment.name,'kind':segment.kind,'structure':segment.structure,
                    'device':'not_run','fixed_axis':0,'plane':'axial','native_acquisition_plane':True,
                    'modeling_source':modeling_source_metadata(session),'reviewed':False}
            report['status']='skipped_absent' if segment.presence=='absent' else 'skipped_empty'
        report.update(slices=entries,accepted_slices=sum(e['accepted'] for e in entries),
                      flagged_slices=sum(not e['accepted'] for e in entries),
                      added_voxels=int(np.count_nonzero((candidate==ident)&(owners!=ident))),
                      removed_voxels=int(np.count_nonzero((candidate!=ident)&(owners==ident))))
        reports.append(report)
    progress(100,'MRI 局部精修候选已完成；需对照原片核对')
    return candidate,reports
