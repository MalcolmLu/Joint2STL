"""MRI absence semantics, native geometry UI, and file export contracts."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication,QFrame,QLabel

from joint2stl.model import Session,Volume,save_project,load_project
from joint2stl.mr_tools import ensure_mr_segments
from joint2stl.meshing import export_stl
from joint2stl.segmentation import paint,initial_segmentation
from joint2stl.recovery import recover_ct_edges
from joint2stl.preprocessing import patient_region
from joint2stl.views import plane_display_title


def example():
    # Native sagittal stack: columns increase P, rows decrease S, slices decrease L.
    affine=np.array([[0,0,-3.9,50],[.5,0,0,-90],[0,-.5,0,25],[0,0,0,1]],float)
    s=Session(Volume(np.full((12,32,40),350,np.int16),affine,modality='MR'),side='未指定')
    ensure_mr_segments(s)
    labels=np.zeros(s.labels.shape,np.uint16);labels[2:8,3:14,3:14]=1
    labels[4:7,18:24,20:24]=6
    s.apply(s.roi,labels,'fixture')
    return s


def test_presence_clear_and_undo_are_one_transaction_and_project_roundtrip(tmp_path):
    s=example();before=s.labels.copy();old_history=len(s.history.undo_stack)
    assert s.update_presence(6,'absent')
    assert not (s.labels==6).any() and s.segment(6).presence=='absent'
    assert len(s.history.undo_stack)==old_history+1
    assert np.array_equal(s.labels==1,before==1)
    s.travel_history();np.testing.assert_array_equal(s.labels,before)
    assert s.segment(6).presence=='unknown'
    s.travel_history(True);assert s.segment(6).presence=='absent'
    path=tmp_path/'mri.j2stl';save_project(s,path);restored=load_project(path)
    assert restored.volume.modality=='MR' and restored.volume.intensity_units=='MR signal'
    assert restored.segment(6).presence=='absent' and restored.segment(6).presence_source=='user_confirmed'
    np.testing.assert_array_equal(restored.volume.values,s.volume.values)
    np.testing.assert_array_equal(restored.labels,s.labels)
    np.testing.assert_array_equal(restored.volume.affine_lps,s.volume.affine_lps)
    with np.load(path,allow_pickle=False) as z:assert json.loads(str(z['metadata']))['schema']==2


def test_old_schema1_ct_project_still_loads(tmp_path):
    v=Volume(np.ones((3,4,5),np.int16),np.eye(4));s=Session(v);seg=s.add_segment('旧股骨')
    labels=np.ones(v.values.shape,np.uint16)
    meta={'schema':1,'joint':'knee','side':'双侧','roi':[[0,n] for n in labels.shape],
          'segments':[{'id':seg.id,'name':seg.name,'color':seg.color,'visible':True,'reviewed':False,'recognition':{}}]}
    path=tmp_path/'old.j2stl'
    with path.open('wb') as f:np.savez_compressed(f,hu=v.values,labels=labels,affine_lps=v.affine_lps,metadata=np.array(json.dumps(meta)))
    loaded=load_project(path)
    assert loaded.volume.modality=='CT' and loaded.segment(1).kind=='bone' and loaded.segment(1).presence=='unknown'


def test_absence_blocks_paint_and_export_omits_acl_but_records_status(tmp_path):
    s=example();ensure_mr_segments(s,acl_absent=True);before=s.labels.copy()
    with pytest.raises(ValueError,match='缺失'):paint(s,'axial',(5,15,15),2,5)
    np.testing.assert_array_equal(s.labels,before)
    s.parameters.update(surface_mm=0,mesh_smoothing='off',compute_backend='cpu')
    directory,manifest=export_stl(s,tmp_path,include_hidden=True,mode='both')
    assert manifest['source_modality']=='MR' and manifest['source_intensity_units']=='MR signal'
    assert manifest['selected_label_ids']==[1,6] and len(list(Path(directory).glob('*.stl')))==3
    acl=next(r for r in manifest['structure_assessments'] if r['structure']=='acl')
    assert acl['presence']=='absent' and acl['presence_source']=='user_stated'
    assert all(x['id']!=5 for x in manifest['segments'])
    assert manifest['combined']['preserves_relative_positions']
    pcl=next(row for row in manifest['segments'] if row['id']==6)
    assert pcl['annotation_coverage']['native_slice_indices_1based']==[5,6,7]
    assert pcl['annotation_coverage']['completeness']=='not_assessed'
    np.testing.assert_array_equal(s.labels,before)


def test_ct_density_entrypoints_reject_mr_without_changes():
    s=example();before=s.labels.copy()
    for call in [lambda:initial_segmentation(s.volume,s.roi),lambda:patient_region(s.volume,s.roi),lambda:recover_ct_edges(s,[1])]:
        with pytest.raises(ValueError,match='MRI'):call()
    np.testing.assert_array_equal(s.labels,before)


def test_native_plane_titles_follow_affine_not_internal_array_key():
    v=example().volume
    assert plane_display_title(v,'axial')=='矢状位 · 采集平面'
    assert plane_display_title(v,'coronal')=='轴位 · 重排'
    assert plane_display_title(v,'sagittal')=='冠状位 · 重排'


@pytest.fixture(scope='module')
def app():
    from joint2stl.__main__ import prepare_qt_plugins
    prepare_qt_plugins()
    return QApplication.instance() or QApplication([])


def test_mri_ui_provides_empty_labels_and_absence_guard_then_restores_ct_controls(app,monkeypatch):
    import joint2stl.app as ui
    class SurfaceStub(QFrame):
        refresh_requested=Signal()
        def __init__(self):
            super().__init__();self.state=QLabel(self);self.caption=QLabel(self);self.actors={};self.bounds=None
            self.vtk=SimpleNamespace(Finalize=lambda:None)
        def set_meshes(self,*_):pass
        def visibility(self,*_):pass
    monkeypatch.setattr(ui,'SurfacePanel',SurfaceStub)
    window=ui.MainWindow();s=example();ensure_mr_segments(s,acl_absent=True)
    window.set_session(s)
    assert window.mri_acl_absent.isChecked() and not window.threshold.isVisibleTo(window)
    assert window.panels['coronal'].isHidden() and window.panels['sagittal'].isHidden()
    assert '矢状位' in window.panels['axial'].title.text()
    assert window.preset.count()==6 and window.rename_button.text()=='选择结构'
    window.active=5;window.refresh_list()
    assert not window.mri_prompt_button.isEnabled() and not window.fill_button.isEnabled()
    monkeypatch.setattr(window,'run_job',lambda *_:pytest.fail('Absent ACL must not start inference'))
    window.mri_prompt('axial',(4,12,12),(4,22,22))
    assert '缺失' in window.status.text()
    from joint2stl.mr_correction import stored_points
    snapshot=s.labels.copy();window.point_action('mr_positive',(4,12,12),'axial')
    assert not stored_points(s)
    window.active=6;window.refresh_list();window.navigate([4,20,22])
    window.point_action('mr_negative',(4,20,22),'coronal');assert not stored_points(s)
    window.point_action('mr_positive',(4,20,22),'axial')
    assert stored_points(s)==[{'label_id':6,'zyx':[4,20,22],'positive':True}]
    np.testing.assert_array_equal(s.labels,snapshot)
    window.mri_presence.setCurrentIndex(window.mri_presence.findData('absent'))
    assert not stored_points(s)  # absence hides hints; undo can restore them
    assert not (s.labels==6).any() and s.segment(6).presence=='absent'
    window.undo(False);assert (s.labels==6).any() and s.segment(6).presence=='unknown'
    assert len(stored_points(s))==1
    window.clear_mri_correction_points();assert not stored_points(s)
    window.mri_review_flags.addItem('旧项目提示',{'label_id':6,'slice_1based':999})
    ct=Session(Volume(np.zeros((8,10,12),np.int16),np.eye(4)))
    window.set_session(ct)
    assert window.mri_review_flags.count()==1 and window.mri_review_flags.currentData() is None
    assert not window.mri_panel.isVisibleTo(window) and window.fill_button.isEnabled()
    assert not window.panels['coronal'].isHidden() and not window.panels['sagittal'].isHidden()
    assert window.rename_button.text()=='命名' and window.threshold.isVisibleTo(window)
    s.dirty=False;ct.dirty=False;window.close()


def test_mri_export_dialog_names_structures_and_keeps_absent_acl_out_of_current_scope(app):
    from joint2stl.app import StlExportDialog
    s=example();ensure_mr_segments(s,acl_absent=True)
    dialog=StlExportDialog(s,5)
    assert '结构' in dialog.scope.button(1).text()
    dialog.scope.button(2).click()
    assert not dialog.action.isEnabled()
    dialog.scope.button(1).click();dialog.mode.button(1).click()
    options=dialog.export_options()
    assert options['ids']==[1,6] and options['combined_name']=='全部结构_组合.stl'
    assert options['mode']=='combined'
    dialog.reject()
