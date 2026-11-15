"""Protect source geometry, absent structures and label ownership during MRI edits."""
import pickle
import sys

import numpy as np
import pytest

from joint2stl.model import Session,Volume,save_project,load_project
from joint2stl.mr_tools import ensure_mr_segments
from joint2stl import mr_correction as correction


def session():
    raw=np.arange(4*26*30,dtype=np.float32).reshape(4,26,30)
    affine=np.array([[0,0,-3.9,80],[.5,0,0,-60],[0,-.5,0,30],[0,0,0,1.]])
    s=Session(Volume(raw,affine,modality='MR'))
    ids=ensure_mr_segments(s,acl_absent=True)
    s.labels[2,9:16,9:16]=ids['femur']
    s.labels[2,12:14,16:19]=ids['tibia']
    s.roi=(slice(1,4),slice(2,24),slice(3,28))
    return s,ids


def fake(monkeypatch,mask):
    calls=[]
    def infer(raw,box,positive,negative=(),device='auto'):
        calls.append((raw.copy(),box,positive,negative))
        return mask.astype(np.float32),{'device':'cpu','inference_performed':True}
    monkeypatch.setattr(correction,'infer_correction',infer)
    return calls


def test_proposal_is_pure_native_single_plane_and_preserves_foreign_owners(monkeypatch):
    s,ids=session();mask=np.zeros(s.labels.shape[1:],bool);mask[8:19,8:21]=True
    calls=fake(monkeypatch,mask);snapshot=pickle.dumps(s)
    proposal,report=correction.propose_correction(s,ids['femur'],2,[(2,11,11)],[(2,10,10)],box=[7,7,22,20])
    assert pickle.dumps(s)==snapshot
    full=s.labels.copy();full[s.roi]=proposal
    np.testing.assert_array_equal(full[0],s.labels[0]);np.testing.assert_array_equal(full[1],s.labels[1]);np.testing.assert_array_equal(full[3],s.labels[3])
    np.testing.assert_array_equal(full==ids['tibia'],s.labels==ids['tibia'])
    assert full[2,10,10]==0 and full[2,11,11]==ids['femur']
    assert report['accepted_slices']==1 and report['slices'][0]['fixed_axis']==0
    np.testing.assert_array_equal(calls[0][0],s.volume.values[2])


@pytest.mark.parametrize('failure',['absent','foreign_positive','offplane_source','outside_roi','boolean_label'])
def test_invalid_inputs_reject_before_inference(monkeypatch,failure):
    s,ids=session()
    monkeypatch.setattr(correction,'infer_correction',lambda *_a,**_k:pytest.fail('Rejected input invoked inference'))
    ident,index,positive=ids['femur'],2,[(2,11,11)]
    if failure=='absent':ident=ids['acl']
    elif failure=='foreign_positive':positive=[(2,12,17)]
    elif failure=='offplane_source':s.volume.affine_lps=np.diag([.5,.5,3.9,1.])
    elif failure=='outside_roi':index=0;positive=[(0,11,11)]
    elif failure=='boolean_label':ident=True
    snapshot=pickle.dumps(s)
    with pytest.raises(ValueError):correction.propose_correction(s,ident,index,positive)
    assert pickle.dumps(s)==snapshot


def test_automatic_large_change_keeps_prior_and_flags_slice(monkeypatch):
    s,ids=session();mask=np.ones(s.labels.shape[1:],bool);fake(monkeypatch,mask)
    local,report=correction.propose_correction(s,ids['femur'],2,box=[4,3,27,23],automatic=True)
    np.testing.assert_array_equal(local,s.labels[s.roi])
    assert not report['slices'][0]['accepted'] and 'large_model_change_needs_review' in report['slices'][0]['flags']


def test_closed_native_bone_hole_fills_but_ligament_hole_does_not(monkeypatch):
    s,ids=session();ring=np.zeros(s.labels.shape[1:],bool);ring[8:19,8:19]=True;ring[11:16,11:16]=False
    fake(monkeypatch,ring)
    s.labels[:]=0;s.labels[2][ring]=ids['femur']
    local,_=correction.propose_correction(s,ids['femur'],2,[(2,9,10)],box=[7,7,20,20])
    assert local[1,12-s.roi[1].start,12-s.roi[2].start]==ids['femur']
    s.labels[s.labels==ids['femur']]=ids['pcl']
    local,_=correction.propose_correction(s,ids['pcl'],2,[(2,9,10)],box=[7,7,20,20])
    assert local[1,12-s.roi[1].start,12-s.roi[2].start]==0


def test_nonfinite_model_output_fails_without_session_edit(monkeypatch):
    s,ids=session();mask=np.full(s.labels.shape[1:],np.nan);fake(monkeypatch,mask)
    snapshot=pickle.dumps(s)
    with pytest.raises(ValueError,match='概率'):correction.propose_correction(s,ids['femur'],2)
    assert pickle.dumps(s)==snapshot


def test_missed_positive_prompt_keeps_original_mask(monkeypatch):
    s,ids=session();mask=np.zeros(s.labels.shape[1:],bool);mask[9:12,9:12]=True;fake(monkeypatch,mask)
    local,report=correction.propose_correction(s,ids['femur'],2,[(2,10,10),(2,15,15)],box=[7,7,20,20])
    np.testing.assert_array_equal(local,s.labels[s.roi])
    assert 'some_foreground_prompts_not_supported' in report['slices'][0]['flags']


def test_automatic_refinement_does_not_drop_an_existing_separate_bone_fragment(monkeypatch):
    s,ids=session();s.labels[:]=0
    s.labels[2,5:12,5:12]=ids['femur'];s.labels[2,16:22,18:24]=ids['femur']
    fake(monkeypatch,s.labels[2]==ids['femur'])
    local,report=correction.propose_correction(s,ids['femur'],2,box=[4,4,26,23],automatic=True)
    np.testing.assert_array_equal(local,s.labels[s.roi])
    assert 'unprompted_existing_fragment_needs_review' in report['slices'][0]['flags']


def test_local_mri_normalization_is_invariant_to_positive_scaling_and_offset():
    pytest.importorskip('torch')
    raw=np.arange(40*48,dtype=np.float64).reshape(40,48)/2000
    a,detail=correction._prepare(raw,[8,9,32,34],[(15,19)],[(29,31)])
    b,other=correction._prepare(raw*.001+7.,[8,9,32,34],[(15,19)],[(29,31)])
    np.testing.assert_allclose(a.numpy(),b.numpy(),atol=2e-6)
    assert detail['crop_xyxy_0based']==other['crop_xyxy_0based']


def test_optional_ai_environment_missing_gives_install_guidance_before_torch_import(monkeypatch):
    monkeypatch.setattr(correction,'correction_model_ready',lambda:False)
    monkeypatch.setitem(sys.modules,'torch',None)
    with pytest.raises(ValueError,match='安装 AI'):
        correction.infer_correction(np.arange(256,dtype=float).reshape(16,16),[2,2,14,14],[(8,8)])


def test_prompt_metadata_roundtrip_and_malformed_metadata_is_ignored(tmp_path):
    s,ids=session();good={'label_id':ids['femur'],'zyx':[2,11,11],'positive':True}
    s.parameters['mr_correction_points']=[good,None,{'label_id':True,'zyx':[2,2,2],'positive':True},
                                         {'label_id':1,'zyx':[999,0,0],'positive':False}]
    assert correction.stored_points(s)==[good]
    path=tmp_path/'mr.j2stl';save_project(s,path);loaded=load_project(path)
    assert correction.stored_points(loaded)==[good]
    np.testing.assert_array_equal(loaded.volume.values,s.volume.values)
    np.testing.assert_array_equal(loaded.volume.affine_lps,s.volume.affine_lps)


def test_batch_uses_original_ownership_and_keeps_pcl_and_acl(monkeypatch):
    s,ids=session();s.labels[2,19:22,20:23]=ids['pcl']
    def infer(raw,box,positive,negative=(),device='auto'):
        return np.ones(raw.shape,np.float32),{'device':'cpu','inference_performed':True}
    monkeypatch.setattr(correction,'infer_correction',infer)
    snapshot=pickle.dumps(s)
    proposal,reports=correction.refine_with_correction(s,[ids['femur'],ids['tibia'],ids['acl']])
    assert pickle.dumps(s)==snapshot
    original=s.labels[s.roi]
    assert not np.any((original==ids['tibia'])&(proposal==ids['femur']))
    assert not np.any((original==ids['femur'])&(proposal==ids['tibia']))
    np.testing.assert_array_equal(proposal==ids['pcl'],original==ids['pcl'])
    assert not np.any(proposal==ids['acl']) and reports[-1]['status']=='skipped_absent'
