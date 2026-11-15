"""Synthetic STL selection and offscreen dialog checks; no clinical/native UI coverage."""

import json
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import numpy as np
import pytest
import trimesh
from PySide6.QtWidgets import QApplication

from joint2stl import meshing
from joint2stl.model import Session, Volume


# These bounds follow directly from voxel-edge coordinates and the fixture's
# reflected, anisotropic affine, independently of Volume.world or the exporter.
EXPECTED_BOUNDS = {
    1: np.array([[37.25, -23.125, 302.25], [39.25, -20.125, 306.75]]),
    2: np.array([[31.75, -17.875, 311.25], [34.25, -14.125, 317.25]]),
    3: np.array([[28.25, -23.125, 303.75], [30.25, -20.125, 308.25]]),
}
# Unsmoothed a*b*c cuboids have 4*(a*b + b*c + c*a) - 4 triangles.
EXPECTED_FACES = {1: 156, 2: 256, 3: 156}


@pytest.fixture
def synthetic_session():
    affine = np.diag([-.5, .75, 1.5, 1.])
    affine[:3, 3] = [40, -25, 300]
    volume = Volume(np.zeros((16, 20, 26), np.int16), affine, 'synthetic export')
    session = Session(volume)
    visible = session.add_segment('可见骨')
    hidden = session.add_segment('隐藏骨')
    other = session.add_segment('未选骨')
    session.add_segment('空骨')
    hidden.visible = False
    labels = np.zeros(volume.hu.shape, np.uint16)
    labels[2:5, 3:7, 2:6] = visible.id
    labels[8:12, 10:15, 12:17] = hidden.id
    labels[3:6, 3:7, 20:24] = other.id
    session.apply(volume.full_box, labels, 'synthetic bones')
    session.parameters.update(surface_mm=0, mesh_smoothing='off', compute_backend='cpu')
    session.dirty = False

    original_labels = session.labels.copy()
    original_hu = volume.hu.copy()
    original_affine = volume.affine_lps.copy()
    original_visibility = [(bone.id, bone.visible) for bone in session.segments]
    original_counts = session.counts().copy()
    original_revision = session.revision
    original_history_version = session.history.version
    yield session
    np.testing.assert_array_equal(session.labels, original_labels)
    np.testing.assert_array_equal(volume.hu, original_hu)
    np.testing.assert_array_equal(volume.affine_lps, original_affine)
    np.testing.assert_array_equal(session.counts(), original_counts)
    assert [(bone.id, bone.visible) for bone in session.segments] == original_visibility
    assert session.revision == original_revision
    assert session.history.version == original_history_version
    assert session.dirty is False


@pytest.fixture(scope='module')
def app():
    from joint2stl.__main__ import prepare_qt_plugins

    prepare_qt_plugins()
    application = QApplication.instance() or QApplication([])
    assert application.platformName() == 'offscreen'
    return application


@pytest.fixture
def make_dialog(app):
    from joint2stl.app import StlExportDialog

    dialogs = []

    def create(session, current_id):
        dialog = StlExportDialog(session, current_id)
        dialogs.append(dialog)
        return dialog

    yield create
    for dialog in dialogs:
        dialog.reject()
        dialog.deleteLater()
    app.processEvents()


@pytest.mark.parametrize('options, expected_ids', [
    ({}, [1, 3]),
    ({'include_hidden': True}, [1, 2, 3]),
    ({'ids': [2], 'include_hidden': True}, [2]),
    ({'ids': [1, 2], 'include_hidden': True}, [1, 2]),
    ({'ids': [1, 2]}, [1]),
    ({'ids': [1, 4], 'include_hidden': True}, [1]),
], ids=['default-visible', 'all-hidden-included', 'hidden-only',
        'explicit-subset', 'hidden-excluded', 'empty-label-skipped'])
def test_export_respects_ids_and_hidden_selection(synthetic_session, tmp_path, options, expected_ids):
    directory, manifest = meshing.export_stl(synthetic_session, tmp_path, **options)
    directory = Path(directory)
    assert manifest['complete']
    assert manifest['export_mode'] == 'separate'
    assert manifest['selection_scope'] == 'checked'
    assert manifest['selected_label_ids'] == expected_ids
    assert [item['id'] for item in manifest['segments']] == expected_ids
    assert manifest['combined'] is None
    filenames = {item['filename'] for item in manifest['files']}
    assert len(filenames) == len(expected_ids)
    assert {file.name for file in directory.glob('*.stl')} == filenames
    for item in manifest['segments']:
        loaded = trimesh.load_mesh(directory / item['filename'], process=True, validate=False)
        assert item['filename'] in filenames
        assert len(loaded.faces) == item['faces'] == EXPECTED_FACES[item['id']]
        np.testing.assert_allclose(loaded.bounds, EXPECTED_BOUNDS[item['id']], rtol=0, atol=1e-5)
        assert loaded.is_watertight and loaded.is_winding_consistent and loaded.volume > 0
    assert json.loads((directory / 'export_manifest.json').read_text(encoding='utf-8')) == manifest


@pytest.mark.parametrize('mode, file_count', [('combined', 1), ('both', 3)])
def test_combined_preserves_lps_shells_and_triangle_counts(synthetic_session, tmp_path, mode, file_count):
    directory, manifest = meshing.export_stl(
        synthetic_session, tmp_path, ids=[1, 2], include_hidden=True,
        mode=mode, combined_name='两块骨.stl',
    )
    directory = Path(directory)
    assert manifest['complete'] and manifest['export_mode'] == mode
    assert manifest['units'] == 'mm' and manifest['coordinates'] == 'DICOM patient LPS'
    assert manifest['selected_label_ids'] == [1, 2]
    assert [item['id'] for item in manifest['segments']] == [1, 2]
    filenames = [item['filename'] for item in manifest['files']]
    assert len(filenames) == len(set(filenames)) == file_count
    assert {file.name for file in directory.glob('*.stl')} == set(filenames)

    combined = manifest['combined']
    assert combined['filename'] == '两块骨.stl'
    assert combined['label_ids'] == [1, 2]
    assert combined['preserves_relative_positions'] is True
    assert combined['boolean_union_applied'] is False
    assert combined['individual_bones_closed'] is True
    loaded = trimesh.load_mesh(directory / combined['filename'], process=True, validate=False)
    raw = trimesh.load_mesh(directory / combined['filename'], process=False)
    assert len(raw.faces) == len(loaded.faces) == combined['faces'] == 412
    assert sum(item['faces'] for item in manifest['segments']) == len(loaded.faces)
    assert loaded.is_watertight and loaded.is_winding_consistent and loaded.volume > 0
    assert np.all(loaded.area_faces > 0) and combined['zero_area_faces'] == 0
    expected_bounds = np.array([[31.75, -23.125, 302.25], [39.25, -14.125, 317.25]])
    np.testing.assert_allclose(loaded.bounds, expected_bounds, rtol=0, atol=1e-5)
    np.testing.assert_allclose(combined['bounds_lps_mm'], expected_bounds, rtol=0, atol=1e-5)

    shells = sorted(loaded.split(only_watertight=False), key=lambda shell: shell.bounds[0, 0])
    assert len(shells) == 2
    for shell, ident in zip(shells, [2, 1]):
        np.testing.assert_allclose(shell.bounds, EXPECTED_BOUNDS[ident], rtol=0, atol=1e-5)
        assert len(shell.faces) == EXPECTED_FACES[ident] and shell.is_watertight
    displacement = shells[0].bounds.mean(axis=0) - shells[1].bounds.mean(axis=0)
    np.testing.assert_allclose(displacement, [-5.25, 5.625, 9.75], rtol=0, atol=1e-5)

    ranges = combined['source_triangle_ranges']
    assert len(ranges) == 2
    first = 0
    for item, source_range in zip(manifest['segments'], ranges):
        ident = item['id']
        assert source_range['id'] == ident
        assert source_range['first_triangle'] == first
        assert source_range['triangle_count'] == item['faces'] == EXPECTED_FACES[ident]
        triangles = raw.triangles[first:first + source_range['triangle_count']]
        points = triangles.reshape(-1, 3)
        np.testing.assert_allclose(
            [points.min(axis=0), points.max(axis=0)], EXPECTED_BOUNDS[ident], rtol=0, atol=1e-5,
        )
        np.testing.assert_allclose(item['bounds_lps_mm'], EXPECTED_BOUNDS[ident], rtol=0, atol=1e-5)
        if mode == 'both':
            individual = trimesh.load_mesh(directory / item['filename'], process=False)
            np.testing.assert_array_equal(triangles, individual.triangles)
        else:
            assert item['filename'] is None
        first += source_range['triangle_count']
    assert first == len(raw.faces)
    assert json.loads((directory / 'export_manifest.json').read_text(encoding='utf-8')) == manifest


@pytest.mark.parametrize('ids, include_hidden', [([], True), ([2], False), ([4], True), ([999], True)])
def test_export_rejects_empty_selection_before_creating_output(synthetic_session, tmp_path, ids, include_hidden):
    parent = tmp_path / 'not-created'
    with pytest.raises(ValueError):
        meshing.export_stl(synthetic_session, parent, ids=ids, include_hidden=include_hidden, mode='both')
    assert not parent.exists()


@pytest.mark.parametrize('mode', ['separate', 'combined', 'both'])
def test_export_rejects_session_without_bones(synthetic_session, tmp_path, mode):
    empty = Session(synthetic_session.volume)
    with pytest.raises(ValueError):
        meshing.export_stl(empty, tmp_path, mode=mode)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('mode', ['', 'invalid', 'BOTH', None])
def test_export_rejects_invalid_mode_without_output(synthetic_session, tmp_path, mode):
    parent = tmp_path / 'not-created'
    with pytest.raises(ValueError):
        meshing.export_stl(synthetic_session, parent, ids=[2], include_hidden=True, mode=mode)
    assert not parent.exists()


def test_export_batch_collisions_preserve_existing_files(synthetic_session, tmp_path, monkeypatch):
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 14, 12, 34, 56, tzinfo=tz)

    monkeypatch.setattr(meshing, 'datetime', FrozenDatetime)
    batch_name = 'Joint2STL_20261114_123456'
    old_files = []
    for suffix in ['', '_2']:
        existing = tmp_path / (batch_name + suffix)
        existing.mkdir()
        sentinel = existing / 'previous.stl'
        sentinel.write_bytes(b'previous synthetic export')
        old_files.append(sentinel)

    first, _ = meshing.export_stl(synthetic_session, tmp_path, ids=[2], include_hidden=True)
    first = Path(first)
    assert first.name == batch_name + '_3'
    first_contents = {file.name: file.read_bytes() for file in first.iterdir()}
    second, _ = meshing.export_stl(synthetic_session, tmp_path, ids=[2], include_hidden=True)
    assert Path(second).name == batch_name + '_4'
    assert {file.name: file.read_bytes() for file in first.iterdir()} == first_contents
    assert all(file.read_bytes() == b'previous synthetic export' for file in old_files)


def test_combined_filename_collision_keeps_individual_geometry(synthetic_session, tmp_path):
    directory, manifest = meshing.export_stl(
        synthetic_session, tmp_path, ids=[1, 2], include_hidden=True,
        mode='both', combined_name='01_可见骨.stl',
    )
    directory = Path(directory)
    filenames = [item['filename'] for item in manifest['files']]
    assert len(filenames) == len(set(filenames)) == 3
    assert {file.name for file in directory.glob('*.stl')} == set(filenames)
    individual = manifest['segments'][0]
    combined = manifest['combined']
    assert individual['filename'] == '01_可见骨.stl'
    assert combined['filename'] != individual['filename']
    single_mesh = trimesh.load_mesh(directory / individual['filename'])
    combined_mesh = trimesh.load_mesh(directory / combined['filename'])
    assert len(single_mesh.faces) == 156 and len(combined_mesh.faces) == 412
    np.testing.assert_allclose(single_mesh.bounds, EXPECTED_BOUNDS[1], rtol=0, atol=1e-5)
    assert json.loads((directory / 'export_manifest.json').read_text(encoding='utf-8')) == manifest


@pytest.mark.parametrize('scope_id, scope, expected_ids, combined_name', [
    (0, 'checked', [1, 3], '选定骨骼_组合.stl'),
    (1, 'all', [1, 2, 3], '全部骨骼_组合.stl'),
    (2, 'current', [2], '选定骨骼_组合.stl'),
])
@pytest.mark.parametrize('mode_id, mode', [(0, 'separate'), (1, 'combined'), (2, 'both')])
def test_dialog_scopes_and_modes_drive_export(
    synthetic_session, make_dialog, tmp_path, scope_id, scope, expected_ids, combined_name, mode_id, mode,
):
    dialog = make_dialog(synthetic_session, 2)
    assert dialog.scope.checkedId() == 0 and dialog.mode.checkedId() == 0
    assert dialog.selected_ids() == [1, 3] and dialog.action.isEnabled()
    dialog.scope.button(scope_id).click()
    dialog.mode.button(mode_id).click()
    assert dialog.action.isEnabled() and dialog.selected_ids() == expected_ids
    options = dialog.export_options()
    assert options == {
        'ids': expected_ids, 'include_hidden': True, 'mode': mode,
        'selection_scope': scope, 'combined_name': combined_name,
    }
    directory, manifest = meshing.export_stl(synthetic_session, tmp_path, **options)
    assert manifest['complete'] and manifest['selected_label_ids'] == expected_ids
    assert manifest['selection_scope'] == scope and manifest['export_mode'] == mode
    file_count = 1 if mode == 'combined' else len(expected_ids) + (mode == 'both')
    assert len(manifest['files']) == len(list(Path(directory).glob('*.stl'))) == file_count


@pytest.mark.parametrize('current_id', [4, 999, None], ids=['empty-label', 'unknown-label', 'no-current'])
def test_dialog_disables_empty_current_and_recovers_on_scope_change(synthetic_session, make_dialog, current_id):
    dialog = make_dialog(synthetic_session, current_id)
    assert dialog.action.isEnabled()
    dialog.scope.button(2).click()
    assert dialog.selected_ids() == [] and not dialog.action.isEnabled()
    dialog.scope.button(1).click()
    assert dialog.selected_ids() == [1, 2, 3] and dialog.action.isEnabled()
    dialog.scope.button(0).click()
    assert dialog.selected_ids() == [1, 3] and dialog.action.isEnabled()


def test_dialog_hidden_bones_remain_available_with_no_checked_bones(synthetic_session, make_dialog):
    session = Session(
        synthetic_session.volume, labels=synthetic_session.labels.copy(),
        segments=[replace(bone, visible=False) for bone in synthetic_session.segments],
    )
    dialog = make_dialog(session, 2)
    assert dialog.selected_ids() == [] and not dialog.action.isEnabled()
    dialog.scope.button(1).click()
    assert dialog.selected_ids() == [1, 2, 3] and dialog.action.isEnabled()
    dialog.scope.button(2).click()
    assert dialog.selected_ids() == [2] and dialog.action.isEnabled()
    assert dialog.export_options()['include_hidden'] is True
    dialog.scope.button(0).click()
    assert dialog.selected_ids() == [] and not dialog.action.isEnabled()
    assert all(not bone.visible for bone in session.segments)
    np.testing.assert_array_equal(session.labels, synthetic_session.labels)


def test_dialog_disables_every_scope_when_all_labels_are_empty(synthetic_session, make_dialog):
    empty = Session(synthetic_session.volume)
    bone = empty.add_segment('空骨')
    dialog = make_dialog(empty, bone.id)
    for scope_id in range(3):
        dialog.scope.button(scope_id).click()
        for mode_id in range(3):
            dialog.mode.button(mode_id).click()
            assert dialog.selected_ids() == []
            assert not dialog.action.isEnabled()
