"""Choose acquired sagittal MRI samples as the modeling source."""
from __future__ import annotations
import numpy as np


def native_acquisition_plane(volume):
    direction=volume.world([1,0,0])-volume.world([0,0,0])
    unit=np.asarray(direction,float)/np.linalg.norm(direction)
    dominant=int(np.argmax(np.abs(unit)))
    return {'plane':('sagittal','coronal','axial')[dominant],
            'sagittal':bool(abs(unit[0])>=.9),
            'normal_lps':unit.tolist(),'oblique':bool(np.abs(unit).max()<.999)}


def sagittal_only(session):
    return getattr(session.volume,'modality','CT')=='MR' and session.parameters.get('mr_sagittal_only',True)


def require_mri_modeling_source(session,plane=None):
    """The internal `axial` key is array axis 0, the acquired plane.

    It is anatomically sagittal only when its normal points along L/R. A
    sagittal reformat of axial data is not an acquired sagittal source. The
    application always enables this policy for MRI; explicit False remains
    available to low-level geometry research callers.
    """
    if not sagittal_only(session):return
    if not native_acquisition_plane(session.volume)['sagittal']:
        raise ValueError('MRI 建模仅使用原始（斜）矢状位序列。请选择矢状采集；当前序列仅供浏览。')
    if plane is not None and plane!='axial':
        raise ValueError('MRI 标注仅允许原始（斜）矢状位采集平面，斜轴位和斜冠状位不参与建模。')


def modeling_source_metadata(session):
    if getattr(session.volume,'modality','CT')!='MR':return None
    native=native_acquisition_plane(session.volume)
    offplane=[]
    for index,report in enumerate(session.ai_reports):
        if report.get('modality')!='MR':continue
        axes={entry.get('fixed_axis',report.get('fixed_axis',0)) for entry in report.get('slices',[])}
        if any(axis!=0 for axis in axes):offplane.append(index)
    return {'policy':'sagittal_acquisition_only' if sagittal_only(session) else 'explicit_unrestricted_research',
            'native_acquisition':native,'source_series_key':getattr(session.volume,'series_key',''),
            'modeling_array_axis':0,'secondary_mpr_inference_enabled':not sagittal_only(session),
            'cross_sequence_fusion':False,'prior_offplane_model_report_indices':offplane,
            'legacy_manual_annotation_provenance':'not recorded before sagittal-source policy'}
