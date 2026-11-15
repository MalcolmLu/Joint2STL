"""Coverage flags never certify anatomy or turn missing masks into diagnoses."""
import pickle
import numpy as np
import pytest

from joint2stl.model import Session,Volume
from joint2stl.mr_tools import ensure_mr_segments
from joint2stl.mr_review import review_native_masks


def make():
    affine=np.array([[0,0,-4,50],[.5,0,0,-70],[0,-.5,0,20],[0,0,0,1]],float)
    s=Session(Volume(np.ones((7,30,40),np.float32),affine,modality='MR'))
    ensure_mr_segments(s,acl_absent=True)
    return s


def test_review_is_pure_and_flags_native_closed_holes_and_partial_ligament():
    s=make();s.labels[3,5:20,5:20]=1;s.labels[3,10:14,10:14]=0
    s.labels[3,22:25,25:28]=6;snapshot=pickle.dumps(s)
    report=review_native_masks(s)
    assert pickle.dumps(s)==snapshot
    assert any(f['code']=='enclosed_bone_holes' and f['slice_1based']==4 for f in report['flags'])
    assert any(f['code']=='single_slice_ligament' and f['label_id']==6 for f in report['flags'])
    assert all(f['label_id']!=5 for f in report['flags'])
    assert not report['reviewed'] and not report['image_model_inference_performed']


def test_empty_structure_does_not_become_absent_and_area_dropout_is_located():
    s=make();s.labels[2,4:20,4:20]=1;s.labels[3,8:11,8:11]=1;s.labels[4,4:20,4:20]=1
    report=review_native_masks(s)
    assert any(f['code']=='abrupt_native_area_change' and f['slice_1based']==4 for f in report['flags'])
    assert any(f['code']=='empty_unassessed_structure' and f['label_id']==6 for f in report['flags'])
    assert s.segment(6).presence=='unknown' and not np.any(s.labels==6)


def test_review_rejects_other_acquisition_sources():
    s=make();s.volume.affine_lps=np.eye(4)
    with pytest.raises(ValueError,match='矢状'):review_native_masks(s)
    s=Session(Volume(np.ones((3,4,5),np.int16),np.eye(4)))
    with pytest.raises(ValueError,match='MR'):review_native_masks(s)
