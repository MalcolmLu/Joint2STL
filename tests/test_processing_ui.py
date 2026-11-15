"""Scope dispatch and RAM guards with synthetic data; no model or CT allocation."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

from types import SimpleNamespace
import numpy as np
import pytest
from PySide6.QtCore import Signal,Qt
from PySide6.QtWidgets import QApplication,QFrame,QLabel

import joint2stl.app as ui
from joint2stl.model import Session,Volume
from joint2stl.resources import SystemMemory,memory_policy

GIB=1024**3


@pytest.fixture(scope='module')
def app():
    from joint2stl.__main__ import prepare_qt_plugins
    prepare_qt_plugins()
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app,monkeypatch):
    class SurfaceStub(QFrame):
        refresh_requested=Signal()
        def __init__(self):
            super().__init__();self.state=QLabel(self);self.actors={};self.bounds=None
            self.vtk=SimpleNamespace(Finalize=lambda:None)
        def set_meshes(self,*args):pass
        def visibility(self,*args):pass
    monkeypatch.setattr(ui,'SurfacePanel',SurfaceStub)
    monkeypatch.setattr(ui,'detect_memory',lambda:SystemMemory(32*GIB,28*GIB))
    window=ui.MainWindow()
    session=Session(Volume(np.full((30,30,30),700,np.int16),np.eye(4)))
    for name in ['股骨','胫骨','腓骨','空骨']:session.add_segment(name)
    labels=np.zeros(session.labels.shape,np.uint16)
    labels[2:9,2:9,2:9]=1;labels[12:19,12:19,12:19]=2;labels[22:28,22:28,22:28]=3
    session.apply(session.roi,labels,'fixture');session.segment(2).visible=False
    window.set_session(session)
    window.active=2;window.refresh_list()
    yield window
    session.dirty=False;window.session.dirty=False;window.close()


def test_multi_checks_drive_model_and_recovery_independently_of_active_radio(window,monkeypatch):
    import joint2stl.ai as ai
    calls=[];errors=[]
    monkeypatch.setattr(ai,'refine_many',lambda session,ids,progress:calls.append(('model',ids)))
    monkeypatch.setattr(ui,'recover_ct_edges',lambda session,ids,progress:calls.append(('recovery',ids)))
    monkeypatch.setattr(window,'run_job',lambda title,fn,finished:fn(lambda *_:None))
    monkeypatch.setattr(window,'show_error',errors.append)
    assert window.active==2 and window.processing_ids()==[1,3]
    assert '2 块骨' in window.ai_button.text()
    window.ai_refine();window.recover_edges()
    window.ai_all.setChecked(False)
    assert window.processing_ids()==[2] and '胫骨' in window.ai_scope_status.text()
    window.ai_refine();window.recover_edges()
    assert calls==[('model',[1,3]),('recovery',[1,3]),('model',[2]),('recovery',[2])]
    assert not errors and not window.session.segment(2).visible


def test_select_all_none_keeps_masks_names_history_and_active_bone(window,monkeypatch):
    session=window.session;labels=session.labels.copy();history=session.history.version
    names=[s.name for s in session.segments]
    errors=[]
    monkeypatch.setattr(window,'show_error',errors.append)
    monkeypatch.setattr(window,'run_job',lambda *args:pytest.fail('Empty selection must not start work'))
    window.check_all_button.click()
    assert window.processing_ids()==[1,2,3] and '3 块骨' in window.ai_button.text()
    window.check_none_button.click()
    assert window.processing_ids()==[] and '0 块骨' in window.ai_button.text()
    window.ai_refine();window.recover_edges()
    assert len(errors)==2 and all('方框' in e for e in errors)
    window.segment_list.item(0).setCheckState(Qt.CheckState.Checked)
    window.segment_list.item(2).setCheckState(Qt.CheckState.Checked)
    assert window.processing_ids()==[1,3]
    assert window.active==2 and session.history.version==history
    assert [s.name for s in session.segments]==names
    np.testing.assert_array_equal(session.labels,labels)


def test_scope_excludes_bones_outside_roi_and_updates_after_edit(window):
    session=window.session
    window.roi_from_view((slice(0,10),slice(0,10),slice(0,10)))
    assert window.processing_ids()==[1] and '1 块骨' in window.ai_button.text()
    session.apply(session.roi,np.zeros((10,10,10),np.uint16),'clear in roi')
    window.refresh_list()
    assert window.processing_ids()==[]
    session.travel_history();window.refresh_list()
    assert window.processing_ids()==[1]
    window.reset_roi()
    assert window.processing_ids()==[1,3]


def test_large_range_checks_use_ram_and_resample_current_availability(monkeypatch):
    errors=[];samples=iter([SystemMemory(16*GIB,12*GIB),SystemMemory(32*GIB,28*GIB),SystemMemory(32*GIB,6*GIB)])
    def refresh():return memory_policy(next(samples))
    controller=SimpleNamespace(session=SimpleNamespace(roi=(slice(0,469),slice(0,512),slice(0,512))),
        refresh_memory_status=refresh,show_error=errors.append)
    assert not ui.MainWindow.memory_check(controller,'model')
    assert ui.MainWindow.memory_check(controller,'model')
    assert not ui.MainWindow.memory_check(controller,'model')
    assert len(errors)==2 and '16.0 GB' in errors[0] and '32.0 GB' in errors[1]


def test_startup_and_preflight_memory_labels_reflect_actual_snapshot(window,monkeypatch):
    assert window.startup_memory.total_bytes==32*GIB and '内存 32 GB' in window.memory_status.text()
    assert '320.0 / 精修 200.0' in window.memory_status.text()
    monkeypatch.setattr(ui,'detect_memory',lambda:SystemMemory(32*GIB,6*GIB))
    window.memory_check('model')
    assert '可用 6.0 GB' in window.memory_status.text()
    assert window.current_memory_policy['operations']['model']['limiting_factor']=='available'
