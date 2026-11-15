"""Mouse routing and reversible removal of an entire mistaken label."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, QSettings, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFrame, QLabel

from joint2stl.model import Session, Volume, Seed, save_project, load_project


def make_session(mr=False):
    affine = (np.array([[0, 0, -3.9, 50], [.5, 0, 0, -90], [0, -.5, 0, 25], [0, 0, 0, 1.]])
              if mr else np.eye(4))
    session = Session(Volume(np.full((9, 48, 56), 700, np.int16), affine, modality='MR' if mr else 'CT'))
    for name, structure in [('股骨', 'femur'), ('胫骨', 'tibia'), ('误识别软组织', '')]:
        session.add_segment(name).structure = structure
    labels = session.labels.copy()
    labels[1:4, 6:12, 6:12] = 1
    labels[2:6, 10:17, 35:42] = 2
    labels[0:2, 2:5, 2:5] = 3
    labels[8, 40:44, 48:52] = 3
    session.apply(session.roi, labels, 'initial')
    return session


def test_delete_clears_all_slices_and_restores_metadata_seeds_points_and_counts(tmp_path):
    session = make_session(True)
    session.segment(3).visible = False
    session.segment(3).reviewed = True
    session.segment(3).recognition = {'name_source': 'manual', 'nested': {'kept': True}}
    session.seeds = [Seed((1, 3, 3), 3), Seed((2, 8, 8), 1)]
    points = [{'label_id': 3, 'zyx': [1, 3, 3], 'positive': True},
              {'label_id': 1, 'zyx': [2, 8, 8], 'positive': False}]
    session.parameters['mr_correction_points'] = deepcopy(points)
    original = session.labels.copy()
    metadata = [asdict(s) for s in session.segments]
    seeds = deepcopy(session.seeds)
    counts = session.counts().copy()
    session.set_roi((slice(1, 5), slice(6, 30), slice(6, 45)))
    session.remember_mesh('old', SimpleNamespace(vertices=np.zeros((3, 3)), faces=np.ones((1, 3))), session.roi)
    assert session.remove_segment(3)
    assert session.segment(3) is None and not np.any(session.labels == 3)
    np.testing.assert_array_equal(session.labels[original != 3], original[original != 3])
    assert (len(session.counts()) <= 3 or session.counts()[3] == 0) and not session._mesh_cache
    assert session.seeds == seeds[1:]
    assert session.parameters['mr_correction_points'] == points[1:]
    saved = tmp_path / 'deleted.j2stl'
    save_project(session, saved)
    restored = load_project(saved)
    assert restored.segment(3) is None and not np.any(restored.labels == 3)
    assert restored.seeds == seeds[1:]
    assert restored.parameters['mr_correction_points'] == points[1:]
    assert session.travel_history() == '删除标签：误识别软组织'
    np.testing.assert_array_equal(session.labels, original)
    np.testing.assert_array_equal(session.counts(), counts)
    assert [asdict(s) for s in session.segments] == metadata
    assert session.seeds == seeds and session.parameters['mr_correction_points'] == points
    assert session.travel_history(True) == '删除标签：误识别软组织'
    assert session.segment(3) is None and not np.any(session.labels == 3)


def test_empty_label_removal_is_undoable_and_does_not_reuse_or_drop_new_labels():
    session = make_session()
    empty = session.add_segment('空的误识别标签')
    assert session.remove_segment(empty.id)
    new = session.add_segment('随后添加的标签')
    assert new.id > empty.id
    session.travel_history()
    assert session.segment(empty.id).name == empty.name and session.segment(new.id) is new
    session.travel_history(True)
    assert session.segment(empty.id) is None and session.segment(new.id) is new


def test_delete_missing_label_leaves_history_and_masks_intact():
    session = make_session()
    labels = session.labels.copy()
    version = session.history.version
    with pytest.raises(ValueError, match='标签'):
        session.remove_segment(99)
    np.testing.assert_array_equal(session.labels, labels)
    assert session.history.version == version


def test_catalogue_allocation_reserves_deleted_id_until_undo_is_unavailable():
    from joint2stl.mr_tools import ensure_mr_segments
    session = make_session(True)
    ids = ensure_mr_segments(session)
    deleted = ids['pcl']
    session.remove_segment(deleted)
    new_ids = ensure_mr_segments(session)
    assert new_ids['pcl'] > deleted
    assert session.segment(new_ids['pcl']).structure == 'pcl'


@pytest.fixture(scope='module')
def app():
    from joint2stl.__main__ import prepare_qt_plugins
    prepare_qt_plugins()
    application = QApplication.instance() or QApplication([])
    application.setStyle('Fusion')
    return application


@pytest.fixture
def window(app, monkeypatch, request, tmp_path):
    import joint2stl.app as ui
    class SurfaceStub(QFrame):
        refresh_requested = Signal()
        def __init__(self):
            super().__init__()
            self.state = QLabel(self)
            self.actors = {}
            self.bounds = None
            self.vtk = SimpleNamespace(Finalize=lambda: None)
        def set_meshes(self, *args): pass
        def visibility(self, *args): pass
    monkeypatch.setattr(ui, 'SurfacePanel', SurfaceStub)
    monkeypatch.setattr(ui, 'QSettings', lambda *_: QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat))
    result = ui.MainWindow()
    result.set_session(make_session(getattr(request, 'param', False)))
    result.show()
    result.activateWindow()
    app.processEvents()
    monkeypatch.setattr(result, 'run_job', lambda title, *args, **kwargs: pytest.fail('Unexpected background task: ' + title))
    yield result
    result.session.dirty = False
    result.close()
    result.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def drag(view, start, end):
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=view.screen_point(start).toPoint())
    QTest.mouseMove(view, view.screen_point(end).toPoint())
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=view.screen_point(end).toPoint())


@pytest.mark.parametrize('window', [False, True], indirect=True)
def test_checked_brush_is_resolved_before_mouse_drag_without_moving_crosshair(window):
    view = window.panels['axial'].view
    window.slide_changed('axial', 4)
    cross = window.cross.copy()
    # Reproduce the actual Cocoa state: selected brush button, stale browse view.
    blocker = QSignalBlocker(window.tool_buttons['paint'])
    window.tool_buttons['paint'].setChecked(True)
    del blocker
    assert window.tool_buttons['paint'].isChecked() and view.tool == 'browse'
    original = window.session.labels.copy()
    drag(view, (4, 24, 22), (4, 24, 26))
    assert view.tool == window.current_tool == 'paint'
    assert '画笔' in window.context.text() and '股骨' in window.context.text()
    assert window.cross == cross and np.all(window.session.labels[4, 24, 22:27] == 1)
    np.testing.assert_array_equal(window.session.labels[:4], original[:4])
    np.testing.assert_array_equal(window.session.labels[5:], original[5:])
    window.undo()
    np.testing.assert_array_equal(window.session.labels, original)


def test_toolbar_eraser_and_shortcuts_keep_tool_target_and_pixels_consistent(window):
    view = window.panels['axial'].view
    window.slide_changed('axial', 2)
    cross = window.cross.copy()
    QTest.mouseClick(window.tool_buttons['erase'], Qt.MouseButton.LeftButton)
    drag(view, (2, 8, 8), (2, 8, 10))
    assert window.session.labels[2, 8, 8] == 0 and window.cross == cross
    other = window.session.labels == 2
    drag(view, (2, 12, 38), (2, 13, 38))
    np.testing.assert_array_equal(window.session.labels == 2, other)
    QTest.keyClick(view, Qt.Key.Key_B)
    drag(view, (2, 24, 22), (2, 24, 26))
    assert window.tool_buttons['paint'].isChecked() and window.session.labels[2, 24, 24] == 1
    window.slide_changed('axial', 3)
    drag(view, (3, 24, 22), (3, 24, 26))
    assert window.active == 1 and window.session.labels[3, 24, 24] == 1


def test_scroll_during_held_brush_cannot_move_stroke_to_another_slice(window):
    view = window.panels['axial'].view
    window.set_tool('paint')
    window.slide_changed('axial', 4)
    point = view.screen_point((4, 24, 22)).toPoint()
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=point)
    wheel = QWheelEvent(QPointF(point), QPointF(view.mapToGlobal(point)), QPoint(), QPoint(0, 120),
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(view, wheel)
    assert window.cross[0] == 4
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=point)
    assert window.session.labels[4, 24, 22] == 1 and window._stroke is None
    window.undo()
    assert window.session.labels[4, 24, 22] == 0


def test_delete_button_updates_list_selection_and_restores_with_undo(window):
    window.segment_list.setCurrentItem(window.segment_list.item(2))
    original = window.session.labels.copy()
    assert window.active == 3 and '误识别软组织' in window.edit_target.text()
    window.delete_segment_button.click()
    assert window.segment_list.count() == 2 and window.session.segment(3) is None
    assert window.active in (1, 2) and not np.any(window.session.labels == 3)
    assert '全部层面' in window.status.text()
    window.undo()
    assert window.active == 3 and window.segment_list.count() == 3
    np.testing.assert_array_equal(window.session.labels, original)
    window.undo(True)
    assert window.session.segment(3) is None and window.active in (1, 2)
    while window.session.segments:
        window.delete_segment_button.click()
    assert window.active == 0 and not window.delete_segment_button.isEnabled()
    assert window.segment_list.count() == 0 and not window.session.labels.any()
    window.undo()
    assert window.active and window.delete_segment_button.isEnabled()


def test_removed_label_actor_is_hidden_while_other_bones_keep_their_visibility():
    from joint2stl.views import SurfacePanel
    class Actor:
        def __init__(self): self.visible = True
        def SetVisibility(self, value): self.visible = bool(value)
    session = make_session()
    surface = SimpleNamespace(actors={i: Actor() for i in (1, 2, 3)},
                              vtk=SimpleNamespace(GetRenderWindow=lambda: SimpleNamespace(Render=lambda: None)))
    session.remove_segment(3)
    SurfacePanel.visibility(surface, session.segments)
    assert surface.actors[1].visible and surface.actors[2].visible and not surface.actors[3].visible
    session.travel_history()
    SurfacePanel.visibility(surface, session.segments)
    assert surface.actors[3].visible
