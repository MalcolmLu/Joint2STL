"""Recover demonstrated missing cortex without dilating joint soft tissue."""
import numpy as np
import pytest

from joint2stl.model import Session,Volume,load_project,save_project
from joint2stl.recovery import recover_ct_edges
from joint2stl.meshing import mesh_from_mask,segment_meshes


def example():
    hu=np.full((40,64,64),35,np.int16)
    truth=np.zeros(hu.shape,bool);truth[6:34,10:34,10:38]=True
    other=np.zeros_like(truth);other[6:34,39:55,10:38]=True
    hu[truth|other]=700
    labels=np.zeros(hu.shape,np.uint16);labels[truth]=1;labels[other]=2
    labels[:,:,34:38][truth[:,:,34:38]]=0
    s=Session(Volume(hu,np.eye(4)),labels=labels)
    s.add_segment('股骨');s.add_segment('胫骨')
    return s,truth,other


def test_ct_recovers_cortex_beyond_model_growth_band_and_preserves_joint_space():
    s,truth,other=example();before=s.labels.copy()
    result,report=recover_ct_edges(s,[1],patient_mask=np.ones(s.labels.shape,bool))
    np.testing.assert_array_equal(result==1,truth)
    np.testing.assert_array_equal(result[other],s.labels[other])
    assert not result[:,34:39].any()  # Actual soft-tissue joint gap remains open.
    np.testing.assert_array_equal(s.labels,before)
    assert report['existing_labels_preserved'] and report['added_voxels']>0


def test_recovery_does_not_grow_into_unseeded_plate_and_has_physical_distance_limit():
    s,truth,_=example()
    s.volume.hu[6:34,10:34,42:46]=900  # Separate high-density support, not a bone marker.
    result,_=recover_ct_edges(s,[1],patient_mask=np.ones(s.labels.shape,bool),max_distance_mm=2)
    assert result[20,20,35]==1 and result[20,20,37]==0
    assert not result[6:34,10:34,42:46].any()


def test_recovery_is_order_independent_and_roi_safe():
    s,_,_=example();s.set_roi((slice(4,36),slice(8,58),slice(8,50)))
    keep=np.ones(s.labels[s.roi].shape,bool)
    a,_=recover_ct_edges(s,[1,2],patient_mask=keep)
    b,_=recover_ct_edges(s,[2,1],patient_mask=keep)
    np.testing.assert_array_equal(a,b)
    old=s.labels.copy();s.apply(s.roi,a,'CT补全')
    outside=np.ones(old.shape,bool);outside[s.roi]=False
    np.testing.assert_array_equal(s.labels[outside],old[outside])
    s.travel_history();np.testing.assert_array_equal(s.labels,old)


def test_repair_records_and_detail_setting_roundtrip(tmp_path):
    s,_,_=example();result,report=recover_ct_edges(s,[1],patient_mask=np.ones(s.labels.shape,bool))
    s.apply(s.roi,result,'CT补全');s.repair_reports.append(report);s.parameters['preserve_detail']=True
    path=tmp_path/'repair.j2stl';save_project(s,path);restored=load_project(path)
    np.testing.assert_array_equal(restored.labels,s.labels)
    assert restored.repair_reports==s.repair_reports and restored.parameters['preserve_detail']


def test_detail_guard_keeps_thin_bone_when_requested_filter_would_erase_it():
    volume=Volume(np.zeros((40,30,30),np.int16),np.diag([.6,.9,1.2,1]))
    mask=np.zeros(volume.hu.shape,bool);mask[5:35,14:16,14:16]=True
    with pytest.raises(ValueError):mesh_from_mask(mask,volume,sigma_mm=1.8,preserve_detail=False)
    mesh=mesh_from_mask(mask,volume,sigma_mm=1.8,preserve_detail=True)
    assert mesh.is_watertight and mesh.volume>0
    assert mesh.metadata['voxel_filter']['applied_sigma_mm']==pytest.approx(.3)
    assert mesh.metadata['voxel_filter']['requested_sigma_mm']==1.8


def test_detail_toggle_invalidates_mesh_cache():
    s,_,_=example();s.parameters['surface_mm']=1.35
    protected=segment_meshes(s,ids=[1])[0][1]
    s.parameters['preserve_detail']=False
    legacy=segment_meshes(s,ids=[1])[0][1]
    assert legacy is not protected
    assert protected.metadata['voxel_filter']['applied_sigma_mm']==.5
    assert legacy.metadata['voxel_filter']['applied_sigma_mm']==1.35


def test_automatic_initial_step_recovers_weak_cortex_before_naming():
    from joint2stl.workflow import prepare_initial
    s,truth,other=example();s.volume.hu[:]=35;s.volume.hu[truth]=700
    s.volume.hu[:,:,34:38][truth[:,:,34:38]]=170
    s=Session(s.volume);parameters=s.parameters.copy();parameters['threshold']=212
    result=prepare_initial(s,parameters)
    assert (result['segmentation'][0][truth]>0).all()
    assert result['repair']['added_voxels']>0
    assert not s.labels.any() and not s.segments


def test_model_batch_applies_ct_recovery_without_mutating_input(monkeypatch):
    import joint2stl.ai as ai
    s,truth,_=example();before=s.labels.copy()
    def proposal(work,ident,*args,**options):
        return work.labels.copy(),{'label_id':ident,'slices':[{'slice_1based':20,'accepted':False}],
                                  'accepted_slices':0,'flagged_slices':1,'device':'cpu'}
    monkeypatch.setattr(ai,'refine_segment',proposal)
    result,reports=ai.refine_many(s,[1])
    np.testing.assert_array_equal(result==1,truth)
    np.testing.assert_array_equal(s.labels,before)
    assert reports[0]['batch_ct_recovery']['input_stage']=='model_output'
    assert reports[0]['slices'][0]['ct_recovered_voxels']>0


def test_enclosed_marrow_is_filled_beyond_outward_growth_limit():
    hu=np.full((50,50,50),35,np.int16);truth=np.zeros(hu.shape,bool);truth[5:45,5:45,5:45]=True;hu[truth]=700
    labels=truth.astype(np.uint16);labels[18:32,7:43,7:43]=0
    s=Session(Volume(hu,np.eye(4)),labels=labels);s.add_segment('股骨')
    result,_=recover_ct_edges(s,[1],patient_mask=np.ones(labels.shape,bool),max_distance_mm=2)
    assert result[25,25,25]==1
    np.testing.assert_array_equal(result==1,truth)


def test_repeated_interior_fill_reaches_stable_result():
    from joint2stl.segmentation import fill_bone_cavities
    rng=np.random.default_rng(47)
    mask=rng.random((22,23,24))>.6
    result=fill_bone_cavities(mask,(1,.8,.7))
    np.testing.assert_array_equal(fill_bone_cavities(result,(1,.8,.7)),result)
