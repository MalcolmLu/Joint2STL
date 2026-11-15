"""Native-slice contour checks; flags are review aids, not correctness labels."""
import numpy as np
from scipy import ndimage as ndi

from .mr_policy import require_mri_modeling_source,modeling_source_metadata

MESSAGES={
    'unresolved_anatomy':'骨块未明确归属，需核对是否为软组织',
    'enclosed_bone_holes':'骨内部有封闭空白，需核对后填充',
    'separate_native_components':'同层有多个分离区域，需核对骨块或多标',
    'abrupt_native_area_change':'与相邻采集层面积变化较大，需核对漏标或多标',
    'single_slice_ligament':'韧带仅标了一层，尚不能确认完整覆盖',
    'empty_unassessed_structure':'结构尚未标注或未评估，不代表缺失',
}


def review_native_masks(session):
    if not session.volume.is_mr:raise ValueError('MRI 轮廓检查仅支持 MR。')
    require_mri_modeling_source(session,'axial')
    flags=[];assessments=[];area_mm2=float(np.prod(session.volume.spacing_zyx[1:]))
    for segment in session.segments:
        if segment.presence=='absent':
            assessments.append({'label_id':segment.id,'state':'absent_user_recorded','presence_source':segment.presence_source});continue
        positions=[];areas=[]
        for z in range(session.roi[0].start,session.roi[0].stop):
            mask=session.labels[z][session.roi[1:]]==segment.id
            area=int(mask.sum());areas.append(area)
            if not area:continue
            positions.append(z)
            codes=[]
            if segment.kind=='bone':
                if segment.structure=='unresolved':codes.append('unresolved_anatomy')
                holes=ndi.binary_fill_holes(mask)&~mask
                # Do not suggest filling another structure's owned pixels.
                holes &= session.labels[z][session.roi[1:]]==0
                if holes.sum()*area_mm2>=1.:codes.append('enclosed_bone_holes')
                cc,n=ndi.label(mask);sizes=np.bincount(cc.ravel())[1:]
                if np.count_nonzero(sizes*area_mm2>=2.)>1:codes.append('separate_native_components')
            for code in codes:flags.append({'label_id':segment.id,'label_name':segment.name,'fixed_axis':0,'slice_1based':z+1,'code':code,'message':MESSAGES[code]})
        if not positions and segment.structure in ('femur','tibia','fibula','patella','acl','pcl'):
            code='empty_unassessed_structure';flags.append({'label_id':segment.id,'label_name':segment.name,'fixed_axis':0,'slice_1based':session.roi[0].start+1,'code':code,'message':MESSAGES[code]})
        if segment.kind=='ligament' and len(positions)==1:
            code='single_slice_ligament';flags.append({'label_id':segment.id,'label_name':segment.name,'fixed_axis':0,'slice_1based':positions[0]+1,'code':code,'message':MESSAGES[code]})
        for i in range(1,len(areas)-1):
            neighbours=[areas[k] for k in (i-1,i+1) if areas[k]*area_mm2>=5.]
            if len(neighbours)!=2:continue
            reference=float(np.median(neighbours))
            if areas[i]<.3*reference or areas[i]>3.3*reference:
                code='abrupt_native_area_change';flags.append({'label_id':segment.id,'label_name':segment.name,'fixed_axis':0,'slice_1based':session.roi[0].start+i+1,'code':code,'message':MESSAGES[code]})
        assessments.append({'label_id':segment.id,'native_slices_with_mask':[z+1 for z in positions],
                            'expert_correctness_assessed':False,'presence':segment.presence,'presence_source':segment.presence_source})
    return {'method':'mr_native_contour_review_v1','modality':'MR','fixed_axis':0,'flags':flags,
            'structure_assessments':assessments,'modeling_source':modeling_source_metadata(session),
            'reviewed':False,'image_model_inference_performed':False,
            'limitations':'Contour/coverage checks only; absence of flags does not establish anatomical correctness.'}
