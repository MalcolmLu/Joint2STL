"""Ownership, physical geometry and SAM transforms; synthetic models, no network."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from joint2stl import ai, mr_bone as bone
from joint2stl.model import Segment, Session, Volume
from joint2stl.mr_tools import ensure_mr_segments


@pytest.fixture(autouse=True)
def prohibit_real_inference_and_ct(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Unexpected real model inference or CT processing')
    monkeypatch.setattr(bone, '_infer_rgb', forbidden)
    for name in ('load_model', 'infer_slice', 'ct_bone_reference', 'screen_proposal',
                 'fill_bone_cavities', 'fill_small_2d_holes', 'refine_segment', 'refine_many'):
        monkeypatch.setattr(ai, name, forbidden)
    bone._load_model_on_device.cache_clear()
    yield
    bone._load_model_on_device.cache_clear()


def make_session(shape=(8, 12, 16), spacing=(2., 1., 1.), roi=None):
    z, y, x = np.indices(shape)
    values = (10000 * z + 100 * y + x).astype(np.float32)
    session = Session(Volume(values, np.diag([*spacing[::-1], 1.]), modality='MR'))
    # Generic coordinate/instance fixtures opt out of the application's
    # sagittal-only source policy; dedicated tests check the default guard.
    session.parameters['mr_sagittal_only']=False
    ids = ensure_mr_segments(session, acl_absent=True)
    session.parameters['min_mm3'] = 0.
    if roi is not None:
        session.set_roi(roi)
    return session, ids


def fake_slices(monkeypatch, session, prediction, *, fallback=False):
    calls = []
    def infer(raw, device='auto'):
        z = int(raw[0, 0] / 10000)
        calls.append((z, device, raw.copy()))
        detail = {'device': 'cpu', 'inference_performed': True,
                  'normalization': {'method': 'synthetic_model_no_ct'}}
        if fallback and len(calls) == 1:
            detail['mps_fallback_reason'] = 'actual synthetic MPS operator failure'
        return prediction[z].copy(), detail
    monkeypatch.setattr(bone, 'infer_bone_slice', infer)
    return calls


def test_candidate_is_pure_and_preserves_ligaments_foreign_orphans_and_outside_roi(monkeypatch):
    roi = (slice(2, 6), slice(2, 10), slice(3, 14))
    session, ids = make_session(roi=roi)
    session.segments.extend([Segment(34, 'foreign', '#ffffff', kind='other'),
                             Segment(35, 'manual bone', '#ff0000', reviewed=True, structure='unresolved')])
    session.labels[0, 3:5, 5:7] = ids['femur']  # outside ROI
    session.labels[5, 9, 13] = ids['tibia']  # stale bone, outside prediction
    session.labels[4, 4, 4] = 35
    session.labels[3, 4, 5] = ids['pcl']
    session.labels[3, 6, 8] = 34
    session.labels[3, 7, 9] = 60000  # orphan, including for global ID allocation
    prediction = np.zeros(session.labels.shape, bool)
    prediction[2:6, 2:9, 3:13] = True
    calls = fake_slices(monkeypatch, session, prediction)
    snapshot = pickle.dumps(session)
    local, report, metadata = bone.prepare_mri_bones(session)
    assert pickle.dumps(session) == snapshot
    assert local.dtype == np.uint16 and local.shape == session.labels[roi].shape
    full = session.labels.copy()
    full[roi] = local
    np.testing.assert_array_equal(full[0], session.labels[0])
    assert full[3, 4, 5] == ids['pcl']
    assert full[3, 6, 8] == 34 and full[3, 7, 9] == 60000
    assert full[5, 9, 13] == 0 and not np.any(full == ids['acl'])
    assert [z for z, _, _ in calls] == list(range(2, 6))
    for z, _, raw in calls:
        np.testing.assert_array_equal(raw, session.volume.values[z])  # full native slices, no ROI resize
    assert report['new_ligament_voxels'] == 0 and report['preserved_nonbone_voxels'] == 3
    assert all(s['kind'] == 'bone' and s['id'] not in (ids['acl'], ids['pcl'], 34) for s in metadata)
    extras = [s for s in metadata if s['structure'] == 'unresolved' and s['id'] != 35]
    assert extras and all(s['id'] > 60000 for s in extras)
    assert next(s for s in metadata if s['id'] == 35)['name'] == 'manual bone'
    generated = np.isin(full, [s['id'] for s in extras])
    assert not np.any(generated & ~prediction)
    assert report['added_voxels'] == int(np.count_nonzero(generated & (session.labels == 0)))


def test_bone_kind_cannot_override_ligament_structure(monkeypatch):
    session, ids = make_session()
    session.segment(ids['pcl']).kind = 'bone'
    session.labels[3, 4, 5] = ids['pcl']
    prediction = np.ones(session.labels.shape, bool)
    fake_slices(monkeypatch, session, prediction)
    candidate, _, metadata = bone.prepare_mri_bones(session)
    assert candidate[3, 4, 5] == ids['pcl']
    assert all(s['id'] != ids['pcl'] for s in metadata)


def test_components_do_not_fuse_across_protected_labels_or_diagonal_contacts(monkeypatch):
    session, ids = make_session()
    prediction = np.zeros(session.labels.shape, bool)
    prediction[2, 4, 4] = prediction[3, 5, 5] = True  # diagonal only
    fake_slices(monkeypatch, session, prediction)
    result, report, metadata = bone.prepare_mri_bones(session)
    assert report['components_retained'] == 2
    assert result[2, 4, 4] != result[3, 5, 5]
    assert np.count_nonzero(result) == 2
    assert len([s for s in metadata if s['structure'] == 'unresolved']) == 2
    assert not np.any(result == ids['acl'])


def test_minimum_component_mm3_uses_affine_determinant_not_spacing_product(monkeypatch):
    session, _ = make_session()
    session.volume.affine_lps = np.array([[1., .5, 0., 0.], [0., 1., 0., 0.],
                                         [0., 0., 3., 0.], [0., 0., 0., 1.]])
    session.parameters['min_mm3'] = 50.
    prediction = np.zeros(session.labels.shape, bool)
    prediction[1:3, 2:4, 2:6] = True  # 16 voxels * det3 = 48 -> drop
    prediction[4:6, 6:8, 8:13] = True  # 20 * det3 = 60 -> retain
    fake_slices(monkeypatch, session, prediction)
    result, report, _ = bone.prepare_mri_bones(session)
    assert not result[1:3, 2:4, 2:6].any()
    assert np.count_nonzero(result) == 20
    assert report['voxel_mm3'] == pytest.approx(3.)
    volumes = sorted(report['component_counts'], key=lambda r: r['mm3'])
    assert [r['mm3'] for r in volumes] == pytest.approx([48., 60.])
    assert [r['retained'] for r in volumes] == [False, True]
    assert report['outer_dilation_mm'] == 0.


def long_bones(monkeypatch, *, reverse_s=False, companions=False, side='双侧'):
    session, ids = make_session((80, 40, 60), (2., 1., 1.))
    session.side = side
    for key, new_id in [('femur', 17), ('tibia', 37), ('fibula', 13), ('patella', 23)]:
        session.segment(ids[key]).id = new_id
        ids[key] = new_id
    if reverse_s:
        session.volume.affine_lps[2, 2] = -2.
        session.volume.affine_lps[2, 3] = 200.
    prediction = np.zeros(session.labels.shape, bool)
    prediction[5:32, 18:26, 24:32] = True
    prediction[45:75, 18:26, 24:32] = True
    if companions:
        prediction[33:40, 2:8, 24:32] = True  # small anterior patella candidate
        prediction[5:25, 27:30, 40:43] = True  # left-lateral lower fibula candidate
        prediction[15:17, 35:38, 8:11] = True  # extra fragment, never forced into a named bone
    fake_slices(monkeypatch, session, prediction)
    return session, ids, prediction


@pytest.mark.parametrize('reverse_s', [False, True])
def test_named_long_bones_follow_patient_superior_not_array_order_and_keep_ids(monkeypatch, reverse_s):
    session, ids, _ = long_bones(monkeypatch, reverse_s=reverse_s)
    session.segment(ids['femur']).name = 'Reviewed manual femur name'
    session.segment(ids['femur']).reviewed = True
    snapshot = pickle.dumps(session)
    candidate, report, metadata = bone.prepare_mri_bones(session)
    assert pickle.dumps(session) == snapshot
    assert candidate[50, 20, 27] == ids['tibia' if reverse_s else 'femur']
    assert candidate[10, 20, 27] == ids['femur' if reverse_s else 'tibia']
    femur = next(s for s in metadata if s['id'] == ids['femur'])
    assert femur['name'] == 'Reviewed manual femur name' and not femur['reviewed']
    assert femur['recognition']['manual_name_preserved']
    assert all(not r['reviewed'] for r in report['components'])


def test_sagittal_native_grid_names_use_lps_not_zyx_axes(monkeypatch):
    session, ids = make_session((12, 80, 40))
    session.volume.affine_lps = np.array([[1., 0., 0., -10.], [0., 0., 2., -10.],
                                         [0., 1., 0., 0.], [0., 0., 0., 1.]])
    prediction = np.zeros(session.labels.shape, bool)
    prediction[3:9, 5:32, 15:23] = True
    prediction[3:9, 45:75, 15:23] = True
    fake_slices(monkeypatch, session, prediction)
    candidate, _, _ = bone.prepare_mri_bones(session)
    assert candidate[5, 10, 18] == ids['tibia']
    assert candidate[5, 50, 18] == ids['femur']


@pytest.mark.parametrize('side, expect_fibula', [('左侧', True), ('双侧', False), ('未指定', False), ('右侧', False)])
def test_small_companion_names_require_unique_geometry_and_respect_known_laterality(monkeypatch, side, expect_fibula):
    session, ids, prediction = long_bones(monkeypatch, companions=True, side=side)
    candidate, report, metadata = bone.prepare_mri_bones(session)
    assert candidate[35, 4, 27] == ids['patella']
    assert (candidate[10, 28, 41] == ids['fibula']) == expect_fibula
    assert candidate[15, 36, 9] > max(ids.values())
    assert all(s['structure'] == 'unresolved' for s in metadata if s['id'] > max(ids.values()))
    assert np.array_equal(candidate != 0, prediction)  # no outer growth or forced filling
    assert report['new_ligament_voxels'] == 0


def test_fused_long_bones_remain_one_unresolved_component(monkeypatch):
    session, ids, prediction = long_bones(monkeypatch)
    prediction[31:46, 18:26, 24:32] = True
    candidate, report, metadata = bone.prepare_mri_bones(session)
    assert report['components_retained'] == 1
    assert not np.any(np.isin(candidate, list(ids.values())))
    assert len([s for s in metadata if s['structure'] == 'unresolved']) == 1


def test_thin_bridge_watershed_separates_named_bones_without_growing_or_editing_other_owners(monkeypatch):
    session, ids, prediction = long_bones(monkeypatch)
    prediction[31:46, 21, 27] = True
    session.set_roi((slice(2, 78), slice(10, 31), slice(15, 45)))
    session.segments.append(Segment(50, 'foreign tissue', '#ffffff', kind='other'))
    session.labels[10, 20, 27] = ids['pcl']
    session.labels[50, 22, 28] = 50
    session.labels[51, 22, 29] = 60000
    session.labels[0, 20, 27] = ids['femur']  # outside ROI
    snapshot = pickle.dumps(session)
    candidate, report, metadata = bone.prepare_mri_bones(session)
    assert pickle.dumps(session) == snapshot
    roi = session.roi
    after = session.labels.copy()
    after[roi] = candidate
    assert after[10, 20, 28] == ids['tibia']
    assert after[50, 20, 27] == ids['femur']
    assert after[10, 20, 27] == ids['pcl']
    assert after[50, 22, 28] == 50 and after[51, 22, 29] == 60000
    assert after[0, 20, 27] == ids['femur']
    assert not np.any(after == ids['acl'])
    editable = (session.labels[roi] == 0) | np.isin(session.labels[roi], ids['femur'])
    generated = np.isin(candidate, [s['id'] for s in metadata])
    assert not np.any(generated & ~(prediction[roi] & editable))
    split = report['instance_separation']
    assert split['method'] == 'physical_core_marker_distance_watershed_v1'
    assert split['core_radius_mm'] == 1.5 and split['core_min_mm3'] == 75.
    assert split['raw_components_split'] == 1
    assert split['watershed_connectivity'] == 26 and split['watershed_line']
    assert split['seam_removed_voxels'] > 0
    assert (split['retained_foreground_voxels'] + split['seam_removed_voxels']
            + split['filtered_small_instance_voxels'] == split['raw_foreground_voxels'])
    assert split['outside_foreground_added_voxels'] == report['new_ligament_voxels'] == 0


def test_core_erosion_uses_physical_radius_on_anisotropic_thin_bridge():
    foreground = np.zeros((17, 90, 90), bool)
    foreground[1:16, 10:80, 3:35] = True
    foreground[1:16, 10:80, 55:87] = True
    # Eight pixels is 2 mm in Y: too narrow for a 1.5 mm eroded core, but a
    # voxel-radius erosion would keep a false connecting core here.
    foreground[8, 41:49, 35:55] = True
    original = foreground.copy()
    spacing = np.array([4., .25, .5])
    instances, report = bone._separate_bone_instances(foreground, spacing, float(np.prod(spacing)))
    np.testing.assert_array_equal(foreground, original)
    assert report['raw_component_count'] == 1 and report['markers_total'] == 2
    assert report['core_marker_count'] == 2 and report['fallback_marker_count'] == 0
    assert instances[8, 45, 20] != instances[8, 45, 70]
    assert instances[8, 45, 20] and instances[8, 45, 70]
    assert report['seam_removed_voxels'] > 0
    assert not np.any((instances != 0) & ~foreground)
    assert np.count_nonzero(instances) + report['seam_removed_voxels'] == foreground.sum()


def test_every_thin_component_without_a_core_gets_one_inside_fallback_marker():
    foreground = np.zeros((7, 30, 40), bool)
    foreground[1:6, 3, 3:20] = True
    foreground[1:6, 12, 3:20] = True
    foreground[1:6, 22, 3:20] = True
    instances, report = bone._separate_bone_instances(foreground, (4., .5, .5), 1.)
    assert report['core_marker_count'] == 0
    assert report['fallback_marker_count'] == report['raw_component_count'] == 3
    assert report['markers_total'] == 3 and report['seam_removed_voxels'] == 0
    np.testing.assert_array_equal(instances != 0, foreground)
    assert len(set(instances[3, [3, 12, 22], 10])) == 3
    assert all(m['maximum_inside_distance_mm'] == .5 for m in report['marker_sources'])


def test_small_eroded_speckles_do_not_create_extra_instance_markers():
    foreground = np.zeros((13, 32, 65), bool)
    foreground[1:12, 4:25, 3:25] = True
    foreground[1:12, 4:25, 40:62] = True
    foreground[6, 14, 25:40] = True
    foreground[5:8, 27:30, 30:33] = True  # 27 mm3 nub, core under 75 mm3
    foreground[6, 14:29, 31] = True
    instances, report = bone._separate_bone_instances(foreground, (1., 1., 1.), 1.)
    assert report['raw_component_count'] == 1
    assert report['markers_total'] == report['core_marker_count'] == 2
    assert set(np.unique(instances)) == {0, 1, 2}
    assert not np.any((instances != 0) & ~foreground)


@pytest.mark.parametrize('filled', [False, True])
def test_empty_and_single_core_roi_preserve_support_without_artificial_seams(filled):
    foreground = np.full((7, 15, 19), filled, bool)
    instances, report = bone._separate_bone_instances(foreground, (2., .5, .5), .5)
    np.testing.assert_array_equal(instances != 0, foreground)
    assert report['markers_total'] == int(filled)
    assert report['seam_removed_voxels'] == 0


def test_elongated_anterior_patella_is_eligible_but_tiny_anterior_fragment_stays_unresolved(monkeypatch):
    session, ids, prediction = long_bones(monkeypatch)
    prediction[:] = False
    prediction[5:32, 22:38, 18:42] = True
    prediction[45:75, 22:38, 18:42] = True
    prediction[29:44, 1:13, 18:38] = True  # S28, LR19, AP11; long-axis S
    prediction[32:36, 16, 27] = True  # anterior speckle cannot compete as patella
    candidate, report, _ = bone.prepare_mri_bones(session)
    assert candidate[35, 5, 27] == ids['patella']
    assert candidate[33, 16, 27] not in ids.values()
    patella = next(r for r in report['components'] if r['structure'] == 'patella')
    assert patella['robust_extents_lps_mm'][2] >= 1.15 * max(patella['robust_extents_lps_mm'][:2])
    assert not patella['reviewed']


def test_two_lateral_rods_with_unknown_side_remain_unresolved(monkeypatch):
    session, ids, prediction = long_bones(monkeypatch, companions=True, side='unknown')
    prediction[5:25, 27:30, 13:16] = True
    snapshot = pickle.dumps(session)
    candidate, report, metadata = bone.prepare_mri_bones(session)
    assert pickle.dumps(session) == snapshot and session.side == 'unknown'
    assert not np.any(candidate == ids['fibula'])
    for point in [(10, 28, 14), (10, 28, 41)]:
        record = next(s for s in metadata if s['id'] == candidate[point])
        assert record['structure'] == 'unresolved'
    assert any('laterality is not inferred' in line.lower() for line in report['limitations'])


def test_uint16_id_exhaustion_is_atomic(monkeypatch):
    session, _ = make_session()
    session.labels[0, 0, 0] = 65535
    prediction = np.zeros(session.labels.shape, bool)
    prediction[2, 4, 4] = True
    fake_slices(monkeypatch, session, prediction)
    snapshot = pickle.dumps(session)
    with pytest.raises(ValueError, match='uint16'):
        bone.prepare_mri_bones(session)
    assert pickle.dumps(session) == snapshot


def test_knee_fov_broad_metaphyses_do_not_require_longitudinal_pca(monkeypatch):
    from joint2stl.model import Volume,Session
    from joint2stl.mr_tools import ensure_mr_segments
    session,ids=make_session(shape=(80,64,100),spacing=(1.,1.,1.))
    session.parameters['min_mm3']=20
    prediction=np.zeros(session.labels.shape,bool)
    prediction[45:75,15:45,15:75]=True
    prediction[5:35,17:46,20:75]=True
    fake_slices(monkeypatch,session,prediction)
    candidate,report,_=bone.prepare_mri_bones(session)
    assert candidate[60,30,40]==ids['femur']
    assert candidate[20,30,40]==ids['tibia']
    assert all(not c['reviewed'] for c in report['components'])


@pytest.mark.parametrize('modality,joint', [('CT', 'knee'), ('MR', 'shoulder'), ('MR', 'leg')])
def test_scope_rejected_before_any_inference(monkeypatch, modality, joint):
    session, _ = make_session()
    session.volume.modality, session.joint = modality, joint
    monkeypatch.setattr(bone, 'infer_bone_slice', lambda *_: pytest.fail('scope validation too late'))
    snapshot = pickle.dumps(session)
    with pytest.raises(ValueError, match='MR knee'):
        bone.prepare_mri_bones(session)
    assert pickle.dumps(session) == snapshot


@pytest.mark.parametrize('minimum', [-1, float('nan'), float('inf'), True])
def test_invalid_physical_threshold_rejected_before_inference(monkeypatch, minimum):
    session, _ = make_session()
    session.parameters['min_mm3'] = minimum
    monkeypatch.setattr(bone, 'infer_bone_slice', lambda *_: pytest.fail('threshold validation too late'))
    with pytest.raises(ValueError, match='mm3'):
        bone.prepare_mri_bones(session)


def test_prediction_failure_and_cancellation_leave_session_unchanged(monkeypatch):
    session, _ = make_session()
    snapshot = pickle.dumps(session)
    def wrong_shape(*_):
        return np.zeros((2, 2), bool), {'device': 'cpu'}
    monkeypatch.setattr(bone, 'infer_bone_slice', wrong_shape)
    with pytest.raises(ValueError, match='native-plane'):
        bone.prepare_mri_bones(session)
    assert pickle.dumps(session) == snapshot
    def cancel(*_):
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        bone.prepare_mri_bones(session, cancel)
    assert pickle.dumps(session) == snapshot


def test_fallback_device_is_used_for_remaining_native_slices(monkeypatch):
    session, _ = make_session()
    calls = fake_slices(monkeypatch, session, np.zeros(session.labels.shape, bool), fallback=True)
    _, report, _ = bone.prepare_mri_bones(session, device='auto')
    assert [requested for _, requested, _ in calls] == ['auto'] + ['cpu'] * (len(calls) - 1)
    assert report['device'] == 'cpu' and len(report['mps_cpu_fallbacks']) == 1


def test_raw_minmax_rgb_has_no_hu_window_and_never_changes_input():
    raw = np.array([[1000., 1050., 1100.], [np.nan, np.inf, -np.inf]])
    before = raw.copy()
    rgb, meta = bone._minmax_rgb(raw)
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_array_equal(rgb[0, :, 0], [0, 127, 255])
    np.testing.assert_array_equal(rgb[..., 0], rgb[..., 1])
    np.testing.assert_array_equal(rgb[1], np.zeros((3, 3), np.uint8))
    assert rgb.dtype == np.uint8 and meta['minimum'] == 1000 and meta['maximum'] == 1100
    assert meta['nonfinite_pixels_replaced'] == 3
    transformed, _ = bone._minmax_rgb(np.array([[20., 40., 60.], [30., 50., 70.]]))
    scaled, _ = bone._minmax_rgb(np.array([[20., 40., 60.], [30., 50., 70.]]) * 7 + 1000)
    np.testing.assert_array_equal(transformed, scaled)


@pytest.mark.parametrize('value', [0., 4000., np.nan, np.inf])
def test_degenerate_native_slices_do_not_invent_bone_or_invoke_model(value):
    result, detail = bone.infer_bone_slice(np.full((3, 5), value))
    assert not result.any() and detail['device'] == 'not_run'
    assert not detail['inference_performed']


def test_sam_long_side_normalization_then_padding_preserves_native_mapping():
    torch = pytest.importorskip('torch')
    raw = np.array([[20., 30., 40., 50.], [40., 50., 60., 70.]])
    rgb, _ = bone._minmax_rgb(raw)
    tensor, size = bone._preprocess_rgb(rgb, 'cpu')
    assert size == (512, 1024) and tensor.shape == (1, 3, 1024, 1024)
    torch.testing.assert_close(tensor[0, :, 0, 0], -torch.tensor(bone.PIXEL_MEAN) / torch.tensor(bone.PIXEL_STD))
    torch.testing.assert_close(tensor[0, :, 511, 1023], (255 - torch.tensor(bone.PIXEL_MEAN)) / torch.tensor(bone.PIXEL_STD))
    assert not tensor[..., 512:, :].any()  # padding is zero AFTER normalization


def test_semantic_mask_one_not_iou_max_and_padding_is_cropped(monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(bone, 'IMAGE_SIZE', 8)
    logits = torch.full((1, 2, 4, 4), -5.)
    logits[:, 0] = 5.  # wrong semantic mask has higher alleged IoU
    logits[:, 1, 2:] = 5.  # mask1 positive ONLY in the padding, not native image
    class Prompt:
        def __call__(self, **kwargs):
            return torch.empty(1, 0, 256), torch.zeros(1, 256, 1, 1)
        def get_dense_pe(self):
            return torch.zeros(1, 256, 1, 1)
    class Fake:
        image_encoder = staticmethod(lambda _: torch.zeros(1, 256, 1, 1))
        prompt_encoder = Prompt()
        mask_decoder = staticmethod(lambda **_: (logits, torch.tensor([[.99, .01]])))
    monkeypatch.setattr(bone, 'load_model', lambda _: (Fake(), 'cpu'))
    # Undo the fixture's inference guard just for this synthetic Torch model.
    monkeypatch.setattr(bone, '_infer_rgb', _REAL_INFER_RGB)
    rgb = np.repeat(np.array([[0, 60, 120, 180], [30, 90, 150, 255]], np.uint8)[..., None], 3, axis=2)
    mask, actual, resized = bone._infer_rgb(rgb, 'cpu')
    assert actual == 'cpu' and resized == [4, 8]
    assert mask.shape == (2, 4) and not mask.any()


_REAL_INFER_RGB = bone._infer_rgb


def test_actual_mps_operator_failure_retries_cpu_and_reports_it(monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(ai, 'torch_device', lambda _: 'mps')
    monkeypatch.setattr(torch.mps, 'empty_cache', lambda: None)
    calls = []
    def infer(rgb, device):
        calls.append(device)
        if device == 'mps':
            raise NotImplementedError('aten::operator is not implemented for MPS')
        return np.ones(rgb.shape[:2], bool), 'cpu', list(rgb.shape[:2])
    monkeypatch.setattr(bone, '_infer_rgb', infer)
    result, detail = bone.infer_bone_slice(np.arange(12).reshape(3, 4))
    assert calls == ['mps', 'cpu'] and result.all()
    assert detail['device'] == 'cpu' and 'MPS' in detail['mps_fallback_reason']


@pytest.mark.parametrize('device,error', [('mps', RuntimeError('shape mismatch')),
                                        ('cpu', RuntimeError('MPS mentioned on CPU')),
                                        ('mps', ValueError('bad output'))])
def test_unrelated_failures_are_not_hidden_by_fallback(monkeypatch, device, error):
    monkeypatch.setattr(ai, 'torch_device', lambda _: device)
    calls = []
    def infer(*args):
        calls.append(args[1])
        raise error
    monkeypatch.setattr(bone, '_infer_rgb', infer)
    with pytest.raises(type(error), match=str(error)):
        bone.infer_bone_slice(np.arange(12).reshape(3, 4))
    assert calls == [device]


def installer_module():
    path = Path(__file__).resolve().parents[1] / 'scripts/setup_mri_model.py'
    spec = importlib.util.spec_from_file_location('bone_installer_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pinned_manifest_loader_and_installer_agree():
    setup = installer_module()
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'models/mri_model_manifest.json').read_text())
    assert manifest['checkpoint_sha256'] == setup.CHECKPOINT_SHA256 == bone.CHECKPOINT_SHA256
    assert manifest['checkpoint_bytes'] == setup.CHECKPOINT_BYTES == bone.CHECKPOINT_BYTES
    assert manifest['checkpoint_source'] == setup.CHECKPOINT_URL == bone.CHECKPOINT_URL
    assert '/5f026279291d2980a01274a0d0aaa6c1a4d68960/' in bone.CHECKPOINT_URL


@pytest.mark.skipif(not bone.mri_bone_model_ready(), reason='Optional pinned local bone weights and Torch/timm not installed')
def test_official_checkpoint_strictly_matches_vendor_and_eval_bias_caches():
    """Exercise the real serialized checkpoint on CPU; no inference or download."""
    torch = pytest.importorskip('torch')
    model, actual = bone.load_model('cpu')
    assert actual == 'cpu' and len(model.state_dict()) == 528
    assert not model.training and model.mask_decoder.num_multimask_outputs == 2
    assert hasattr(model, 'attention_fusion')
    adapters = [name for name, _ in model.named_modules() if name.endswith('MLP_Adapter')]
    assert len(adapters) == 10  # 8 TinyViT block adapters and 2 decoder adapters
    caches = [module for module in model.modules() if hasattr(module, 'attention_bias_idxs')]
    assert caches
    for module in caches:
        torch.testing.assert_close(module.ab, module.attention_biases[:, module.attention_bias_idxs], rtol=0, atol=0)


def test_installer_reuses_valid_local_file_without_network(monkeypatch, tmp_path):
    setup = installer_module()
    path = tmp_path / 'bone_sam.pth'
    path.write_bytes(b'pinned')
    monkeypatch.setattr(setup, 'CHECKPOINT_BYTES', 6)
    monkeypatch.setattr(setup, 'CHECKPOINT_SHA256', hashlib.sha256(b'pinned').hexdigest())
    monkeypatch.setattr(setup.urllib.request, 'urlopen', lambda *_a, **_k: pytest.fail('redownloaded pinned file'))
    assert setup.install(path) == path
    path.write_bytes(b'broken')
    with pytest.raises(ValueError, match='not overwritten'):
        setup.install(path)
    assert path.read_bytes() == b'broken'


@pytest.mark.parametrize('payload,declared,valid', [(b'12345678', None, True),
                                                (b'123456789', None, False),
                                                (b'123', None, False),
                                                (b'XXXXXXXX', None, False),
                                                (b'12345678', '99999', False)])
def test_installer_download_is_byte_bounded_hashed_and_atomic(monkeypatch, tmp_path, payload, declared, valid):
    setup = installer_module()
    monkeypatch.setattr(setup, 'CHECKPOINT_BYTES', 8)
    monkeypatch.setattr(setup, 'CHECKPOINT_SHA256', hashlib.sha256(b'12345678').hexdigest())
    requests, reads = [], []
    class Response(io.BytesIO):
        headers = {} if declared is None else {'Content-Length': declared}
        def read(self, n=-1):
            reads.append(n)
            return super().read(n)
    def open_url(url, **kwargs):
        requests.append(url)
        return Response(payload)
    monkeypatch.setattr(setup.urllib.request, 'urlopen', open_url)
    path = tmp_path / 'bone_sam.pth'
    if valid:
        assert setup.install(path) == path and path.read_bytes() == b'12345678'
    else:
        with pytest.raises(ValueError):
            setup.install(path)
        assert not path.exists()
    assert requests == [bone.CHECKPOINT_URL]
    assert all(0 < n <= 9 for n in reads)
    assert not list(tmp_path.glob('*.part'))


@pytest.mark.parametrize('requested', ['mps', 'cuda'])
def test_loader_is_weights_only_strict_and_limits_device_before_to(monkeypatch, tmp_path, requested):
    torch = pytest.importorskip('torch')
    pytest.importorskip('timm')
    from joint2stl.vendor.segment_any_bone import build
    path = tmp_path / 'checkpoint'
    path.write_bytes(b'fixed')
    monkeypatch.setattr(bone, 'checkpoint_path', lambda: path)
    monkeypatch.setattr(bone, 'CHECKPOINT_BYTES', 5)
    monkeypatch.setattr(bone, 'CHECKPOINT_SHA256', hashlib.sha256(b'fixed').hexdigest())
    events = []
    class Fake:
        def load_state_dict(self, state, *, strict):
            assert strict and list(state) == ['weight']
            events.append('strict_load')
        def to(self, device):
            assert 'fraction' in events and 'strict_load' in events
            events.append(('to', device))
            return self
        def eval(self):
            assert events[-1] == ('to', requested)
            events.append('eval_refresh')
            return self
    def safe_load(path, *, map_location, weights_only):
        assert weights_only is True and map_location == 'cpu'
        return {'weight': torch.zeros(1)}
    monkeypatch.setattr(build, 'build_bone_model', Fake)
    monkeypatch.setattr(torch, 'load', safe_load)
    monkeypatch.setattr(torch, 'set_num_threads', lambda n: events.append(('threads', n)))
    monkeypatch.setattr(torch.mps, 'set_per_process_memory_fraction', lambda n: events.append('fraction') if n == .35 else pytest.fail('wrong MPS fraction'))
    monkeypatch.setattr(torch.cuda, 'set_per_process_memory_fraction', lambda n: events.append('fraction') if n == .5 else pytest.fail('wrong CUDA fraction'))
    monkeypatch.setattr(torch.backends.cuda, 'matmul', SimpleNamespace(allow_tf32=True))
    monkeypatch.setattr(torch.backends, 'cudnn', SimpleNamespace(allow_tf32=True))
    _, device = bone._load_model_on_device(requested)
    assert device == requested and events[-1] == 'eval_refresh'
    if requested == 'cuda':
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    assert any(e[0] == 'threads' and e[1] <= 4 for e in events if isinstance(e, tuple))
