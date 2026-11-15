"""Regressions for cross-study selection, label ownership, and slab boundaries."""
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

from joint2stl.model import Session,Volume,save_project,load_project
from joint2stl.long_scan import prepare_whole_leg, whole_leg_plan
from joint2stl.resources import SystemMemory, memory_policy
from joint2stl.segmentation import install_initial
from joint2stl.dicom_io import scan_directory,study_recommendations
from test_pipeline import write_ct


def test_each_study_recommends_thin_soft_but_no_cross_study_default(tmp_path, monkeypatch):
    from pydicom.uid import generate_uid
    from PySide6.QtWidgets import QApplication
    from joint2stl.app import SeriesDialog
    monkeypatch.setenv('QT_QPA_PLATFORM','offscreen')
    app=QApplication.instance() or QApplication([])
    good=[]
    for index,dz in enumerate((.625,.8)):
        study=generate_uid(); soft=generate_uid(); sharp=generate_uid();good.append(soft)
        for z in range(3):
            write_ct(tmp_path/f's{index}-{z}.dcm',soft,study,z*dz,kernel='STANDARD',thickness=dz)
            write_ct(tmp_path/f'b{index}-{z}.dcm',sharp,study,z*.5,kernel='BONE',thickness=.5)
    series,unread=scan_directory(tmp_path)
    assert len({x.study_key for x in series})==2 and not any(x.recommended for x in series)
    recommendation=study_recommendations(series)
    assert len(recommendation)==2 and all(x.kernel=='STANDARD' for x in recommendation.values())
    dlg=SeriesDialog(series,unread,None)
    assert not dlg.open_button.isEnabled() and dlg.choices.checkedId()==-1
    row=next(i for i,x in enumerate(dlg.series) if x.spacing[2]>.7)
    dlg.choice_buttons[row].click();assert dlg.open_button.isEnabled()
    dlg.accept_selected();assert dlg.selected is series[row]


def test_whole_leg_matches_monolithic_components_including_diagonal_seam(tmp_path):
    hu=np.full((21,25,23),-1000,np.int16)
    # Each five-layer piece is too small for the global size threshold. Join
    # this shaft across slabs even when the next occupied voxel is diagonal.
    expected=np.zeros(hu.shape,bool)
    for z in range(3,18):
        y,x=6+2*((z-3)//5),6+2*((z-3)//5)
        hu[z,y:y+2,x:x+2]=700;expected[z,y:y+2,x:x+2]=True
    hu[11:13,18:20,18:20]=700 # genuine isolated fragment, below threshold
    s=Session(Volume(hu,np.diag([.7,.8,1.,1.])),joint='leg')
    s.set_roi((slice(3,19),slice(3,23),slice(4,22)))
    s.parameters.update(remove_table=False,fill=False,min_mm3=25)
    progress=[]
    result=prepare_whole_leg(s,s.parameters,lambda p,_:progress.append(p),slab_depth=5)
    ids=install_initial(s,result['segmentation'],result['recognition'])
    assert ids==[1] and progress[-1]==100 and progress==sorted(progress)
    assert result['preprocessing']['labels_crossing_slab_boundaries']==[1]
    np.testing.assert_array_equal(s.labels!=0,expected)
    path=tmp_path/'leg.j2stl';save_project(s,path);read=load_project(path)
    assert read.joint=='leg' and read.roi==s.roi
    np.testing.assert_array_equal(read.labels,s.labels)
    np.testing.assert_array_equal(read.volume.hu,hu)


def test_whole_leg_slab_budget_does_not_relax_normal_operation_limits():
    gib=1024**3
    roi=(slice(0,1609),slice(0,512),slice(0,512))
    low=memory_policy(SystemMemory(32*gib,5*gib))
    with pytest.raises(ValueError,match='内存不足'):whole_leg_plan(roi,low)
    high=memory_policy(SystemMemory(32*gib,15*gib))
    plan=whole_leg_plan(roi,high)
    assert plan['full_voxels'] > high['operations']['initial']['limit_voxels'] > plan['max_slab_voxels']
    assert 2<=plan['slab_depth']<=80 and plan['max_slab_voxels']<=20_000_000


def _adjacency_project():
    hu=np.full((24,24,26),50,np.int16)
    mask=np.zeros(hu.shape,np.uint16)
    mask[4:20,5:19,3:12]=1
    mask[4:20,5:19,13:22]=2
    hu[4:20,5:19,3:22]=700
    s=Session(Volume(hu,np.eye(4)),labels=mask)
    for name in ('胫骨','腓骨'):s.add_segment(name)
    return s


def test_recovery_keeps_reviewed_gap_and_can_still_restore_elsewhere(tmp_path):
    from joint2stl.recovery import recover_ct_edges
    s=_adjacency_project()
    s.labels[12,12,4]=0
    keep=np.ones(s.labels.shape,bool)
    base,_=recover_ct_edges(s,[1,2],patient_mask=keep)
    assert base[10,10,12]!=0
    for segment in s.segments:
        segment.recognition={'method':'source_ct_review_and_local_seeded_split','original_label_id':1}
    s.repair_reports.append({'method':'right_proximal_tibiofibular_seam_review_v1','original_label_id':1,
        'label_ids':[1,2],'seam_local_edit_bounding_box_zyx_0based':[[4,20],[5,19],[12,13]],
        'segmented_bones_do_not_touch_26_neighborhood':True})
    out=tmp_path/'protected.j2stl';save_project(s,out);s=load_project(out)
    a,report=recover_ct_edges(s,[1,2],patient_mask=keep)
    assert report['protected_reviewed_gap_voxels'] and not a[4:20,5:19,12].any() and a[12,12,4]==1
    assert not np.any((a==1)&ndi.binary_dilation(a==2,structure=ndi.generate_binary_structure(3,3)))


def test_model_candidate_cannot_claim_original_vacated_neighbor(monkeypatch):
    from joint2stl import ai
    s=_adjacency_project()
    s.parameters.update(recover_edges=False, fill=False)
    voxel=(12,12,11)
    monkeypatch.setattr(ai,'load_model',lambda *_:(None,'cpu'))
    def predict(image,*_):return np.ones(image.shape,np.float32),.9,'cpu'
    monkeypatch.setattr(ai,'infer_slice',predict)
    def fake_screen(prior,probability,ct,spacing,foreign,**kwargs):
        pred=prior.copy()
        if prior[12,10]:pred[12,11]=False
        else:pred[12,11:14]=True
        return pred,{'accepted':True,'area_ratio_to_initial':1.0,'flags':[]}
    monkeypatch.setattr(ai,'screen_proposal',fake_screen)
    before=s.labels.copy()
    candidate,_=ai.refine_many(s,[1,2],device='cpu')
    assert candidate[voxel]==0  # the preceding edit cleared it; no later bone can occupy it
    np.testing.assert_array_equal(s.labels,before)
    assert not np.any((before!=0)&(candidate!=0)&(candidate!=before))


def test_model_batch_cleanup_honors_original_owners():
    from joint2stl.recovery import recover_ct_edges
    s=_adjacency_project()
    before=s.labels.copy()
    # Lower-density shared row previously belonged to the unselected tibia;
    # the fibular boundary grows into it if only the batch mask is considered.
    candidate=before.copy();candidate[4:20,5:19,11]=0
    # A thin dense extension on the fibular side reaches the vacated voxel;
    # the remaining tibia cannot compete across this non-dense strip.
    s.volume.hu[4:20,5:19,10]=50
    keep=np.ones(candidate.shape,bool)
    free,_=recover_ct_edges(s,[2],labels=candidate,patient_mask=keep)
    guarded,_=recover_ct_edges(s,[2],labels=candidate,patient_mask=keep,original_owners=before)
    assert (free[4:20,5:19,11]==2).any()
    assert not guarded[4:20,5:19,11].any()


def test_long_outer_track_is_removed_without_dropping_deep_shaft_or_foot():
    from joint2stl.long_scan import peripheral_rail_candidates
    v=Volume(np.zeros((2,2,2),np.int16),np.eye(4))
    eligible=np.array([False,True,True,True,True,True,True])
    totals=np.array([0,3000,4000,3500,400,150,100]); anchors=np.array([0,2800,3900,800,20,10,80]); unknown=np.zeros(7,int)
    lower=np.array([[0,0,0],[10,24,28],[50,25,87],[0,69,5],[10,71,6],[110,25,26],[10,82,26]])
    upper=np.array([[0,0,0],[385,61,66],[405,60,130],[380,79,14],[35,76,17],[118,34,32],[20,86,31]])
    assert np.flatnonzero(peripheral_rail_candidates(v,eligible,totals,anchors,unknown,lower,upper)).tolist()==[3,4]
