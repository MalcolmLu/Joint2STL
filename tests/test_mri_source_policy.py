"""MRI modeling takes acquired sagittal samples, independent of UI key names."""
from pathlib import Path
import numpy as np
import pytest
from joint2stl.model import Volume,Session,save_project,load_project
from joint2stl.mr_policy import native_acquisition_plane,require_mri_modeling_source,modeling_source_metadata
from joint2stl.mr_tools import ensure_mr_segments,predict_mr_prompt,refine_mr_many
from joint2stl.mr_bone import prepare_mri_bones
from joint2stl.segmentation import paint,clean_segment
from joint2stl.meshing import export_stl


def session(sagittal=True):
    affine=(np.array([[0,0,-3.9,50],[.5,0,0,-90],[0,-.5,0,25],[0,0,0,1]],float)
            if sagittal else np.diag([.5,.5,3.9,1]))
    s=Session(Volume(np.full((8,32,40),150,np.float32),affine,modality='MR'))
    ensure_mr_segments(s,acl_absent=True)
    return s


def test_default_policy_is_sagittal_and_survives_project_roundtrip(tmp_path):
    s=session();assert s.parameters['mr_sagittal_only'] is True
    require_mri_modeling_source(s,'axial')  # acquired array plane, anatomically sagittal
    assert native_acquisition_plane(s.volume)['sagittal']
    path=tmp_path/'sag.j2stl';save_project(s,path);r=load_project(path)
    assert r.parameters['mr_sagittal_only'] is True
    meta=modeling_source_metadata(r)
    assert meta['policy']=='sagittal_acquisition_only' and not meta['secondary_mpr_inference_enabled']


@pytest.mark.parametrize('plane',['coronal','sagittal'])
def test_secondary_reformat_cannot_generate_or_paint_voxels(plane,monkeypatch):
    import joint2stl.ai as ai
    s=session();old=s.labels.copy()
    monkeypatch.setattr(ai,'infer_slice',lambda *a,**kw:pytest.fail('Reformat reached inference'))
    with pytest.raises(ValueError,match='矢状'):
        predict_mr_prompt(s,6,plane,(3,10,10),(3,20,20))
    with pytest.raises(ValueError,match='矢状'):paint(s,plane,(3,15,15),2,1)
    np.testing.assert_array_equal(s.labels,old)


@pytest.mark.parametrize('operation',['bone','refine','mesh'])
def test_axial_source_is_not_sagittal_merely_because_view_is_reformatted(operation,tmp_path,monkeypatch):
    import joint2stl.mr_bone as bone
    s=session(False);old=s.labels.copy()
    monkeypatch.setattr(bone,'infer_bone_slice',lambda *_:pytest.fail('Non-sag source reached model'))
    with pytest.raises(ValueError,match='矢状'):
        if operation=='bone':prepare_mri_bones(s)
        elif operation=='refine':refine_mr_many(s,[1])
        else:export_stl(s,tmp_path)
    np.testing.assert_array_equal(s.labels,old)
    assert not list(tmp_path.iterdir())


def test_export_records_sagittal_source_and_rejects_legacy_offplane_results(tmp_path):
    s=session();labels=s.labels.copy();labels[2:6,8:20,8:20]=1;s.apply(s.roi,labels,'sag mask')
    s.parameters.update(surface_mm=0,mesh_smoothing='off',compute_backend='cpu')
    folder,manifest=export_stl(s,tmp_path)
    assert manifest['modeling_source']['native_acquisition']['plane']=='sagittal'
    assert manifest['modeling_source']['modeling_array_axis']==0
    s.ai_reports.append({'modality':'MR','fixed_axis':1,'slices':[{'fixed_axis':1,'accepted':True}]})
    before=list(tmp_path.iterdir())
    with pytest.raises(ValueError,match='旧模型'):export_stl(s,tmp_path)
    assert list(tmp_path.iterdir())==before


def test_mri_fill_uses_only_acquired_contours_not_secondary_plane_cavities():
    s=session();labels=s.labels.copy()
    # Channel open on every acquired sagittal layer: filling a secondary
    # orthogonal plane could close it, while acquired-plane fill must retain it.
    labels[1:7,8:24,8:28]=1;labels[3:5,8:20,14:20]=0
    s.apply(s.roi,labels,'open sagittal channel')
    result=clean_segment(s,1,'fill')
    assert not result[3:5,8:20,14:20].any()
    assert not np.any(result==5)
