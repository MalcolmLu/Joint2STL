"""Offscreen component checks; these do not claim macOS native interaction coverage."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF
from joint2stl.model import Session,Volume
from joint2stl.views import SliceView

@pytest.fixture(scope='module')
def app():
    from joint2stl.__main__ import prepare_qt_plugins
    prepare_qt_plugins()
    return QApplication.instance() or QApplication([])

@pytest.mark.parametrize('plane',['axial','coronal','sagittal'])
@pytest.mark.parametrize('reversed_z',[False,True])
def test_screen_voxel_roundtrip_in_all_planes(app,plane,reversed_z):
    a=np.diag([.6,.8,-1.2 if reversed_z else 1.2,1.])
    s=Session(Volume(np.zeros((31,42,57),np.int16),a))
    view=SliceView(plane);view.resize(510,360);view.session=s;view.cross=[15,20,28]
    for point in [(15,4,5),(15,35,40)] if plane=='axial' else [(5,20,10),(25,20,45)] if plane=='coronal' else [(5,10,28),(25,30,28)]:
        screen=view.screen_point(point)
        assert view.from_position(screen)==point
    assert view.from_position(QPointF(-10,-10)) is None
    view.close()


def test_edit_keeps_accessible_list_items_alive_and_tools_match_state(app,monkeypatch):
    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import QFrame,QLabel
    import joint2stl.app as ui
    class RenderStub:
        def Finalize(self):pass
    class SurfaceStub(QFrame):
        refresh_requested=Signal()
        def __init__(self):
            super().__init__();self.state=QLabel(self);self.actors={};self.bounds=None;self.vtk=RenderStub()
        def set_meshes(self,*args):pass
        def visibility(self,*args):pass
    monkeypatch.setattr(ui,'SurfacePanel',SurfaceStub)
    window=ui.MainWindow()
    s=Session(Volume(np.ones((31,42,57),np.int16)*800,np.eye(4)))
    bone=s.add_segment('股骨')
    s.apply((slice(0,31),slice(0,42),slice(0,57)),np.ones(s.labels.shape,np.uint16),'initial')
    window.set_session(s)
    item=window.segment_list.item(0);icon=item.icon().cacheKey()
    window.tool_buttons['erase'].setChecked(True)
    assert all(p.view.tool=='erase' for p in window.panels.values())
    window.begin_stroke('axial',15);window.stroke_point('axial',(15,20,28));window.end_stroke()
    assert window.segment_list.item(0) is item and item.icon().cacheKey()==icon
    assert s.labels[15,20,28]==0
    window.undo(False);assert s.labels[15,20,28]==bone.id
    assert window.segment_list.item(0) is item
    window.roi_spins[0][0].setValue(20)
    window.roi_spins[0][1].setValue(10)
    assert window.roi_spins[0][1].value()==31
    assert s.roi[0]==slice(19,31)
    s.dirty=False;window.close()


def test_committed_save_result_is_applied_before_pending_close(monkeypatch):
    from types import SimpleNamespace
    import joint2stl.app as ui
    events=[]
    monkeypatch.setattr(ui,'QTimer',SimpleNamespace(singleShot=lambda ms,fn:events.append('close scheduled')))
    controller=SimpleNamespace(_on_result=lambda result:events.append(('saved',result)),
        pending_close=True,_apply_on_close=True,finish_job=lambda:events.append('job finished'),close=lambda:None)
    ui.MainWindow.job_success(controller,'project.j2stl')
    assert events==['job finished',('saved','project.j2stl'),'close scheduled']


def test_failed_save_keeps_dirty_project_open(monkeypatch):
    from types import SimpleNamespace
    import joint2stl.app as ui
    events=[]
    monkeypatch.setattr(ui,'QTimer',SimpleNamespace(singleShot=lambda *args:events.append('unexpected close')))
    controller=SimpleNamespace(_on_result=None,pending_close=True,_apply_on_close=True,
        finish_job=lambda:None,show_error=lambda text:events.append(text),close=lambda:None)
    ui.MainWindow.job_failed(controller,'disk full')
    assert controller.pending_close is False and events==['disk full']


def test_menu_selector_keeps_full_values_and_blocks_rebuild_navigation(app):
    from joint2stl.controls import MenuSelector
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QPushButton
    selector=MenuSelector();selector.resize(90,32)
    long_name='轴位 271 · 与初分割差异较大，需要核对'
    selector.addItem('选择异常层',None);selector.addItem(long_name,271)
    events=[];selector.currentIndexChanged.connect(events.append)
    def choose():
        dialog=selector._dialog
        try:
            assert dialog.isVisible()
            next(b for b in dialog.findChildren(QPushButton) if b.text()==long_name).click()
        finally:
            if dialog.isVisible():dialog.reject()
    QTimer.singleShot(0,choose);selector.click()
    assert selector.currentData()==271 and selector.currentText()==long_name
    assert selector.accessibleName()==long_name and events==[1]
    selector.blockSignals(True);selector.clear();selector.addItem('重建后的条目',99);selector.blockSignals(False)
    assert selector.currentData()==99 and events==[1]
    selector.close()


def test_import_dialog_selects_soft_thin_even_if_bone_row_is_first(app):
    from joint2stl.app import SeriesDialog
    from joint2stl.dicom_io import Series
    bone=Series('bone','302','1.0 bone',files=['a']*3,spacing=(.7,.7,1),thickness=1,affine=np.eye(4),kernel='B_VSHARP_C',reconstruction_kind='sharp')
    soft=Series('soft','304','1.0 soft',files=['b']*3,spacing=(.7,.7,1),thickness=1,affine=np.eye(4),kernel='B_SOFT_C',reconstruction_kind='soft')
    dialog=SeriesDialog([bone,soft],0,None)
    assert dialog.selected_index==1 and dialog.choice_buttons[1].isChecked() and dialog.open_button.isEnabled()
    dialog.reject()
    only_bone=SeriesDialog([bone],0,None)
    assert only_bone.selected_index==-1 and not only_bone.open_button.isEnabled()
    only_bone.choice_buttons[0].click()
    assert only_bone.selected_index==0 and only_bone.open_button.isEnabled()
    only_bone.reject()


def test_numeric_controls_ignore_scroll_but_keep_deliberate_steps(app):
    from joint2stl.controls import PrecisionSpinBox,PrecisionDoubleSpinBox
    from PySide6.QtCore import QPoint,Qt
    from PySide6.QtGui import QWheelEvent
    for widget,value in [(PrecisionSpinBox(),212),(PrecisionDoubleSpinBox(),1.35)]:
        widget.setRange(0,5000);widget.setValue(value);widget.show();widget.setFocus()
        event=QWheelEvent(QPointF(5,5),QPointF(5,5),QPoint(0,0),QPoint(0,120),
                          Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.NoScrollPhase,False)
        app.sendEvent(widget,event)
        assert widget.value()==value
        widget.stepUp();assert widget.value()>value
        widget.close()
