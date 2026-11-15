"""Synthetic geometry/ownership tests; no model downloads or clinical assertions."""
from itertools import product
import pickle

import numpy as np
import pytest
from scipy import ndimage as ndi

from joint2stl import ai, mr_tools as mr
from joint2stl.model import Session, Volume


@pytest.fixture(autouse=True)
def no_real_inference_or_ct_processing(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Unexpected real inference/model loading or CT processing')

    for name in ('load_model', 'infer_slice', 'ct_bone_reference', 'screen_proposal',
                 'fill_bone_cavities', 'fill_small_2d_holes', 'refine_segment', 'refine_many'):
        monkeypatch.setattr(ai, name, forbidden)
    from joint2stl import preprocessing, recovery, segmentation
    monkeypatch.setattr(recovery, 'recover_ct_edges', forbidden)
    monkeypatch.setattr(preprocessing, 'patient_region', forbidden)
    monkeypatch.setattr(segmentation, 'fill_bone_cavities', forbidden)
    monkeypatch.setattr(segmentation, 'fill_small_2d_holes', forbidden)


def session(shape=(7, 40, 44), spacing=(2., 1., 1.), roi=None):
    z, y, x = np.indices(shape)
    signal = (z * 167 + y * 7 + x * 1.25 - 900).astype(np.float32)
    affine = np.diag([*spacing[::-1], 1.])
    s = Session(Volume(signal, affine, modality='MR'))
    # Explicitly exercise the generic lower-level coordinate engine here;
    # production sagittal-only guards are tested separately.
    s.parameters['mr_sagittal_only']=False
    ids = mr.ensure_mr_segments(s)
    if roi is not None:
        s.set_roi(roi)
    return s, ids


def plane_view(array, axis, index):
    slices = [slice(None)] * 3
    slices[axis] = index
    return array[tuple(slices)]


def central_infer(image, box, device='auto', window=None):
    x0, y0, x1, y1 = box
    y, x = (y0 + y1) // 2, (x0 + x1) // 2
    probability = np.zeros(image.shape, np.float32)
    probability[y:y + 2, x:x + 2] = .5
    return probability, .37, 'cpu'


def assert_ui_report(report, axis, plane):
    assert {'label_id', 'device', 'accepted_slices', 'flagged_slices', 'slices'} <= report.keys()
    assert report['fixed_axis'] == axis and report['plane'] == plane
    assert isinstance(report['device'], str)  # the UI calls .upper(), even for skips
    assert report['accepted_slices'] + report['flagged_slices'] == len(report['slices'])
    assert report['reviewed'] is False
    assert report['model_score_calibration'] == 'UNCALIBRATED'
    for entry in report['slices']:
        assert {'slice_1based', 'accepted', 'flags', 'model_score_uncalibrated'} <= entry.keys()
        assert entry['fixed_axis'] == axis and entry['plane'] == plane
        assert entry['slice_1based'] == entry['fixed_index_0based'] + 1
        assert entry['reviewed'] is False


def test_catalogue_defaults_idempotence_and_no_invented_voxels():
    s, ids = session()
    assert mr.MR_STRUCTURES == [
        ('femur', '股骨', 'bone'), ('tibia', '胫骨', 'bone'),
        ('fibula', '腓骨', 'bone'), ('patella', '髌骨', 'bone'),
        ('acl', '前交叉韧带 ACL', 'ligament'), ('pcl', '后交叉韧带 PCL', 'ligament')]
    for key, name, kind in mr.MR_STRUCTURES:
        segment = s.segment(ids[key])
        assert (segment.name, segment.kind, segment.structure) == (name, kind, key)
        assert segment.presence == 'unknown' and segment.presence_source == ''
        assert not segment.reviewed
    snapshot = pickle.dumps(s)
    assert mr.ensure_mr_segments(s) == ids
    assert pickle.dumps(s) == snapshot
    assert not s.labels.any()


def test_ensure_reuses_exact_unassigned_names_and_does_not_use_orphan_ids():
    s = Session(Volume(np.zeros((3, 4, 5), np.float32), np.eye(4), modality='MR'))
    prior = s.add_segment('前交叉韧带 ACL')
    prior.reviewed = True
    prior.presence, prior.presence_source = 'present', 'user_confirmed'
    s.labels[1, 1, 1] = 14
    ids = mr.ensure_mr_segments(s)
    assert ids['acl'] == prior.id
    assert prior.kind == 'ligament' and prior.structure == 'acl' and prior.reviewed
    assert prior.presence == 'present'
    assert all(ident > 14 for key, ident in ids.items() if key != 'acl')
    assert s.labels[1, 1, 1] == 14


@pytest.mark.parametrize('bad', [1, 0, 'false', 'true', None, np.bool_(True)])
def test_absence_requires_an_explicit_boolean_without_partial_setup(bad):
    s = Session(Volume(np.zeros((3, 4, 5), np.float32), np.eye(4), modality='MR'))
    snapshot = pickle.dumps(s)
    with pytest.raises(ValueError, match='bool'):
        mr.ensure_mr_segments(s, acl_absent=bad)
    assert pickle.dumps(s) == snapshot


def test_explicit_absence_clears_global_acl_and_has_atomic_metadata_mask_undo():
    s, ids = session(roi=(slice(2, 5), slice(4, 30), slice(3, 32)))
    acl = s.segment(ids['acl'])
    acl.presence, acl.presence_source = 'present', 'user_confirmed'
    acl.recognition = {'user_note': 'retain'}
    s.labels[0, 1, 1] = acl.id  # outside the editing ROI, still cleared by presence API
    s.labels[3, 10:14, 12:16] = acl.id
    s.labels[3, 11, 13] = ids['pcl']
    before = s.labels.copy()
    mr.ensure_mr_segments(s, acl_absent=True)
    assert (acl.presence, acl.presence_source) == ('absent', 'user_stated')
    assert not (s.labels == acl.id).any()
    assert s.labels[3, 11, 13] == ids['pcl']
    assert acl.recognition == {'user_note': 'retain'}
    assert s.travel_history()
    np.testing.assert_array_equal(s.labels, before)
    assert (acl.presence, acl.presence_source) == ('present', 'user_confirmed')
    assert s.travel_history(redo=True)
    snapshot = pickle.dumps(s)
    mr.ensure_mr_segments(s, acl_absent=False)
    assert pickle.dumps(s) == snapshot


def test_uint16_capacity_failure_does_not_partially_add_labels():
    s = Session(Volume(np.zeros((3, 4, 5), np.float32), np.eye(4), modality='MR'))
    s.labels[1, 1, 1] = 65534
    snapshot = pickle.dumps(s)
    with pytest.raises(ValueError, match='uint16'):
        mr.ensure_mr_segments(s)
    assert pickle.dumps(s) == snapshot


@pytest.mark.parametrize('api', ['ensure', 'prompt', 'refine'])
def test_all_entry_points_require_mr_before_inference(api):
    s = Session(Volume(np.zeros((4, 8, 9), np.float32), np.eye(4)))
    ident = s.add_segment().id
    snapshot = pickle.dumps(s)
    with pytest.raises(ValueError, match='MR volume'):
        if api == 'ensure':
            mr.ensure_mr_segments(s)
        elif api == 'prompt':
            mr.predict_mr_prompt(s, ident, 'axial', (2, 2, 2), (2, 6, 7))
        else:
            mr.refine_mr_many(s, [ident])
    assert pickle.dumps(s) == snapshot


@pytest.mark.parametrize('key', ['acl', 'pcl'])
def test_absent_prompt_raises_before_model_even_with_stale_voxels(key):
    s, ids = session()
    segment = s.segment(ids[key])
    segment.presence, segment.presence_source = 'absent', 'user_stated'
    s.labels[3, 12:18, 15:21] = segment.id
    snapshot = pickle.dumps(s)
    with pytest.raises(ValueError, match='absent'):
        mr.predict_mr_prompt(s, segment.id, 'axial', (3, 8, 10), (3, 28, 30))
    assert pickle.dumps(s) == snapshot


@pytest.mark.parametrize('plane,axis,fixed', [('axial', 0, 4), ('coronal', 1, 8), ('sagittal', 2, 11)])
def test_prompt_geometry_raw_orientation_roi_and_lps_corners(monkeypatch, plane, axis, fixed):
    s, ids = session(shape=(10, 18, 22), roi=(slice(1, 9), slice(2, 16), slice(3, 20)))
    # Reflected, oblique voxel axes; bounding only two diagonal corners is wrong.
    s.volume.affine_lps = np.array([[0., .8, 1.7, 12.], [-.6, 0., 0., -22.],
                                    [0., 1.2, -.8, 40.], [0., 0., 0., 1.]])
    start, end = [2, 4, 5], [7, 13, 17]
    start[axis] = end[axis] = fixed
    axes = [a for a in range(3) if a != axis]
    box = [start[axes[1]], start[axes[0]], end[axes[1]], end[axes[0]]]
    calls = []

    def infer(image, got_box, device, *, window):
        calls.append((image.copy(), got_box, device, window))
        p, score, actual = central_infer(image, got_box, device, window)
        p[0:2, 0:2] = .99  # an unrelated component, outside the prompt
        return p, score, actual

    monkeypatch.setattr(ai, 'infer_slice', infer)
    snapshot = pickle.dumps(s)
    candidate, report = mr.predict_mr_prompt(s, ids['acl'], plane, end, start, device='mps')
    assert pickle.dumps(s) == snapshot
    assert candidate.dtype == np.uint16 and candidate.shape == s.labels[s.roi].shape
    assert not np.shares_memory(candidate, s.labels)
    assert len(calls) == 1
    raw = np.take(s.volume.values, fixed, axis=axis)
    low, high = np.percentile(raw, [1, 99])
    np.testing.assert_array_equal(calls[0][0], np.clip(raw, low, high).astype(np.float32))
    assert calls[0][1] == box and calls[0][2] == 'mps'
    np.testing.assert_allclose(calls[0][3], [low, high])
    full = s.labels.copy()
    full[s.roi] = candidate
    coords = np.argwhere(full == ids['acl'])
    assert len(coords) == 4 and np.all(coords[:, axis] == fixed)
    assert report['fixed_indices_0based'] == [fixed]
    assert report['accepted_slices'] == 1 and report['device'] == 'cpu'
    assert report['box_xyxy_0based'] == box
    assert_ui_report(report, axis, plane)
    entry = report['slices'][0]
    lo, hi = np.zeros(3), np.zeros(3)
    lo[axis], hi[axis] = fixed, fixed + 1
    lo[axes], hi[axes] = [box[1], box[0]], [box[3], box[2]]
    corners = np.array(list(product(*zip(lo - .5, hi - .5))))
    world = corners[:, ::-1] @ s.volume.affine_lps[:3, :3].T + s.volume.affine_lps[:3, 3]
    np.testing.assert_allclose(entry['box_corners_lps_mm'], world)
    np.testing.assert_allclose(entry['bounds_lps_mm'], [world.min(0), world.max(0)])
    assert s.segment(ids['acl']).presence == 'unknown'


def test_explicit_span_is_bounded_by_roi_and_records_every_coronal_index(monkeypatch):
    s, ids = session(shape=(9, 14, 20), roi=(slice(1, 8), slice(4, 10), slice(2, 18)))
    monkeypatch.setattr(ai, 'infer_slice', central_infer)
    candidate, report = mr.predict_mr_prompt(s, ids['femur'], 'coronal', (2, 5, 4), (7, 5, 16), span=4)
    assert report['fixed_indices_0based'] == [4, 5, 6, 7, 8, 9]
    assert report['accepted_slices'] == 6 and report['span'] == 4
    assert np.all((candidate == ids['femur']).any(axis=(0, 2)))
    assert_ui_report(report, 1, 'coronal')


@pytest.mark.parametrize('kwargs,match', [
    ({'plane': 'transverse'}, 'plane'),
    ({'end_zyx': (4, 24, 28)}, 'same fixed'),
    ({'start_zyx': (3., 8, 10)}, 'integer triples'),
    ({'start_zyx': (3, 8)}, 'integer triples'),
    ({'start_zyx': (-1, 8, 10)}, 'inside'),
    ({'end_zyx': (3, 24, 44)}, 'inside'),
    ({'end_zyx': (3, 24, 11)}, '2 pixels'),
    ({'end_zyx': (3, 9, 28)}, '2 pixels'),
    ({'span': -1}, 'span'), ({'span': 5}, 'span'),
    ({'span': 1.0}, 'span'), ({'span': True}, 'span'),
    ({'ident': 0}, 'existing'), ({'ident': 9999}, 'existing'),
])
def test_invalid_prompts_fail_before_model_without_input_changes(kwargs, match):
    s, ids = session()
    arguments = dict(ident=ids['femur'], plane='axial', start_zyx=(3, 8, 10), end_zyx=(3, 24, 28))
    arguments.update(kwargs)
    snapshot = pickle.dumps(s)
    with pytest.raises(ValueError, match=match):
        mr.predict_mr_prompt(s, **arguments)
    assert pickle.dumps(s) == snapshot


def test_clipped_roi_box_minimum_and_outside_slice_are_checked_before_inference():
    s, ids = session(roi=(slice(2, 5), slice(10, 30), slice(10, 32)))
    with pytest.raises(ValueError, match='2 pixels'):
        mr.predict_mr_prompt(s, ids['femur'], 'axial', (3, 4, 2), (3, 11, 20))
    with pytest.raises(ValueError, match='outside the ROI'):
        mr.predict_mr_prompt(s, ids['femur'], 'axial', (1, 12, 12), (1, 24, 28))


def test_two_pixel_box_is_valid_and_threshold_is_half(monkeypatch):
    s, ids = session()

    def infer(image, box, device, *, window):
        p = np.zeros(image.shape, np.float32)
        p[12, 14] = .5
        p[12, 15] = np.nextafter(np.float32(.5), np.float32(0))
        return p, .11, 'cpu'

    monkeypatch.setattr(ai, 'infer_slice', infer)
    candidate, report = mr.predict_mr_prompt(s, ids['acl'], 'axial', (3, 12, 14), (3, 14, 16))
    assert candidate[3, 12, 14] == ids['acl'] and not candidate[3, 12, 15]
    assert np.count_nonzero(candidate) == 1 and report['accepted_slices'] == 1


def test_prompt_clips_box_to_roi_and_preserves_foreign_labels_and_unedited_slices(monkeypatch):
    s, ids = session(roi=(slice(1, 6), slice(8, 32), slice(10, 36)))
    s.labels[0, 15, 20] = ids['femur']
    s.labels[3, 20, 23] = 59999  # orphan labels are foreign too
    s.labels[4, 20, 23] = ids['pcl']
    s.labels[3, 3, 3] = ids['acl']  # target outside ROI
    before = s.labels.copy()
    calls = []

    def infer(image, box, device, *, window):
        calls.append(box)
        return central_infer(image, box, device, window)

    monkeypatch.setattr(ai, 'infer_slice', infer)
    candidate, report = mr.predict_mr_prompt(s, ids['acl'], 'axial', (3, 2, 3), (3, 35, 40))
    assert calls == [[10, 8, 36, 32]]
    full = before.copy()
    full[s.roi] = candidate
    assert full[3, 20, 23] == 59999
    mask = before != 0
    np.testing.assert_array_equal(full[mask], before[mask])
    changed = np.argwhere(full != before)
    assert len(changed) == 3 and np.all(changed[:, 0] == 3)
    assert report['removed_voxels'] == 0 and report['added_voxels'] == 3


def test_prompt_support_respects_two_mm_with_anisotropic_pixels(monkeypatch):
    s, ids = session(spacing=(3., .6, 2.5))
    s.labels[3, 28, 30] = ids['acl']  # current target outside the prompt/halo
    s.labels[3, 15, 18] = ids['pcl']

    def infer(image, box, device, *, window):
        p = np.zeros(image.shape, np.float32)
        p[12:18, 12:18] = .9
        p[2:28, 15] = .9
        p[15, 2:30] = .9
        return p, .92, 'mps'

    monkeypatch.setattr(ai, 'infer_slice', infer)
    candidate, report = mr.predict_mr_prompt(s, ids['acl'], 'axial', (3, 10, 10), (3, 20, 20))
    box_mask = np.zeros((40, 44), bool)
    box_mask[10:20, 10:20] = True
    distance = ndi.distance_transform_edt(~box_mask, sampling=(.6, 2.5))
    added = (candidate[3] == ids['acl']) & (s.labels[3] == 0)
    assert added.any() and np.max(distance[added]) <= 2.
    assert candidate[3, 7, 15] == ids['acl']
    assert not candidate[3, 6, 15] and not candidate[3, 15, 9]
    assert candidate[3, 28, 30] == ids['acl'] and candidate[3, 15, 18] == ids['pcl']
    assert 'touches_bbox' in report['slices'][0]['flags']
    assert report['warning_slices'] == 1 and report['flagged_slices'] == 0


def test_prompt_can_remove_only_current_target_within_edit_scope(monkeypatch):
    s, ids = session()
    s.labels[3, 12:22, 14:24] = ids['femur']
    s.labels[3, 3, 3] = ids['femur']
    s.labels[3, 16, 18] = ids['tibia']

    def infer(image, box, device, *, window):
        p = np.zeros(image.shape, np.float32)
        p[14:22, 14:24] = .9
        return p, .5, 'cpu'

    monkeypatch.setattr(ai, 'infer_slice', infer)
    candidate, report = mr.predict_mr_prompt(s, ids['femur'], 'axial', (3, 10, 12), (3, 26, 30))
    assert not candidate[3, 12:14, 14:24].any()
    assert candidate[3, 3, 3] == ids['femur'] and candidate[3, 16, 18] == ids['tibia']
    assert report['removed_voxels'] == 20


@pytest.mark.parametrize('output', ['empty', 'unrelated', 'nonfinite', 'overlarge'])
def test_unusable_prompt_outputs_never_fabricate_a_structure(monkeypatch, output):
    s, ids = session()

    def infer(image, box, device, *, window):
        p = np.zeros(image.shape, np.float32)
        if output == 'unrelated':
            p[0:2, 0:2] = .99
        elif output == 'nonfinite':
            p[18, 20] = np.nan
        elif output == 'overlarge':
            p[:] = .99
        return p, .99, 'cpu'

    monkeypatch.setattr(ai, 'infer_slice', infer)
    candidate, report = mr.predict_mr_prompt(s, ids['pcl'], 'axial', (3, 10, 12), (3, 28, 32))
    assert not candidate.any() and report['accepted_slices'] == 0 and report['flagged_slices'] == 1
    if output == 'overlarge':
        assert {'overlarge_prediction', 'touches_bbox'} <= set(report['slices'][0]['flags'])
    assert s.segment(ids['pcl']).presence == 'unknown'


def test_prior_component_selection_does_not_switch_to_a_different_central_island(monkeypatch):
    s, ids = session()
    s.labels[3, 11:14, 13:16] = ids['acl']

    def infer(image, box, device, *, window):
        p = np.zeros(image.shape, np.float32)
        p[11:14, 13:16] = .7
        p[18:23, 23:28] = .99  # larger, central, but not connected to the prior
        return p, .9, 'cpu'

    monkeypatch.setattr(ai, 'infer_slice', infer)
    candidate, report = mr.predict_mr_prompt(s, ids['acl'], 'axial', (3, 8, 10), (3, 28, 32))
    np.testing.assert_array_equal(candidate, s.labels)
    assert report['accepted_slices'] == 1


def test_mri_percentile_window_sanitizes_nonfinite_outliers_without_hu(monkeypatch):
    s, ids = session()
    raw = s.volume.values[3]
    raw[0, :5] = [np.nan, np.inf, -np.inf, -1e20, 1e20]
    raw_copy = raw.copy()
    expected_window = np.percentile(raw[np.isfinite(raw)], [1, 99])
    calls = []

    def infer(image, box, device, *, window):
        assert np.isfinite(image).all()
        calls.append(window)
        return central_infer(image, box, device, window)

    monkeypatch.setattr(ai, 'infer_slice', infer)
    _, report = mr.predict_mr_prompt(s, ids['femur'], 'axial', (3, 10, 12), (3, 28, 32))
    np.testing.assert_allclose(calls, [expected_window])
    np.testing.assert_array_equal(s.volume.values[3], raw_copy)
    assert report['slices'][0]['normalization']['nonfinite_pixels_replaced'] == 3
    assert report['intensity_units'] == 'MR signal'
    assert 'UNCALIBRATED' in report['limitations']
    assert not any('hu' in key.lower() for key in report)


@pytest.mark.parametrize('constant', [0., -9000., 450.])
def test_constant_signal_has_a_finite_nonzero_normalization_window(monkeypatch, constant):
    s, ids = session()
    s.volume.values[:] = constant

    def infer(image, box, device, *, window):
        assert np.isfinite(window).all() and window[0] < constant < window[1]
        assert np.all(image == constant)
        return central_infer(image, box, device, window)

    monkeypatch.setattr(ai, 'infer_slice', infer)
    _, report = mr.predict_mr_prompt(s, ids['acl'], 'axial', (3, 10, 12), (3, 28, 32))
    assert report['slices'][0]['normalization']['degenerate_percentiles']


def test_no_finite_signal_raises_before_inference_without_input_changes():
    s, ids = session()
    s.volume.values[3] = np.nan
    snapshot = pickle.dumps(s)
    with pytest.raises(ValueError, match='no finite'):
        mr.predict_mr_prompt(s, ids['acl'], 'axial', (3, 10, 12), (3, 28, 32))
    assert pickle.dumps(s) == snapshot


def test_nonfinite_score_is_explicit_and_json_safe(monkeypatch):
    s, ids = session()

    def infer(*args, **kwargs):
        p, _, device = central_infer(*args, **kwargs)
        return p, np.nan, device

    monkeypatch.setattr(ai, 'infer_slice', infer)
    _, report = mr.predict_mr_prompt(s, ids['acl'], 'axial', (3, 10, 12), (3, 28, 32))
    entry = report['slices'][0]
    assert entry['model_score_uncalibrated'] is None
    assert 'nonfinite_model_score' in entry['flags']
    import json
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('failure', ['inference', 'shape', 'progress'])
def test_multislice_prompt_failure_is_atomic(monkeypatch, failure):
    s, ids = session()
    count = 0

    def infer(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            if failure == 'inference':
                raise RuntimeError('cancelled model')
            if failure == 'shape':
                return np.zeros((2, 2)), .7, 'cpu'
        return central_infer(*args, **kwargs)

    def progress(value, text):
        if failure == 'progress' and value > 0:
            raise RuntimeError('cancelled progress')

    monkeypatch.setattr(ai, 'infer_slice', infer)
    snapshot = pickle.dumps(s)
    with pytest.raises((RuntimeError, ValueError)):
        mr.predict_mr_prompt(s, ids['femur'], 'axial', (3, 10, 12), (3, 28, 32), progress, span=1)
    assert pickle.dumps(s) == snapshot


@pytest.mark.parametrize('key,growth', [('acl', 1.), ('femur', 2.)])
def test_refinement_native_planes_and_physical_growth_caps(monkeypatch, key, growth):
    roi = (slice(1, 4), slice(4, 60), slice(5, 68))
    s, ids = session(shape=(5, 64, 72), spacing=(3., .6, .4), roi=roi)
    ident = ids[key]
    s.labels[2, 20:44, 22:50] = ident
    prior = s.labels[2] == ident
    distance = ndi.distance_transform_edt(~prior, sampling=(.6, .4))
    seen = []

    def infer(image, box, device, *, window):
        seen.append((image.shape, box, device))
        return (distance <= growth + .4).astype(np.float32) * .9, .23, 'cuda'

    monkeypatch.setattr(ai, 'infer_slice', infer)
    snapshot = pickle.dumps(s)
    candidate, reports = mr.refine_mr_many(s, [ident], device='cuda')
    assert pickle.dumps(s) == snapshot
    full = s.labels.copy()
    full[roi] = candidate
    added = (full[2] == ident) & ~prior
    assert added.any() and distance[added].max() <= growth + 1e-9
    np.testing.assert_array_equal(full[2] == ident, distance <= growth + 1e-9)
    assert seen[0][0] == (64, 72) and seen[0][2] == 'cuda'
    report = reports[0]
    assert_ui_report(report, 0, 'axial')
    assert report['fixed_indices_0based'] == [2] and report['accepted_slices'] == 1
    assert report['max_model_growth_mm'] == growth
    assert report['slices'][0]['max_model_growth_mm'] == growth
    assert s.segment(ident).presence == 'unknown'


def test_refinement_keeps_model_holes_and_can_remove_bright_signal_prior(monkeypatch):
    s, ids = session()
    ident = ids['femur']
    s.volume.values[:] = 10000  # not a density rule for retaining this synthetic prior
    s.labels[3, 10:30, 12:32] = ident
    s.labels[3, 18:21, 20:23] = 0
    proposal = s.labels[3] == ident
    proposal[10:12] = False

    def infer(image, box, device, *, window):
        return proposal.astype(np.float32) * .9, .8, 'cpu'

    monkeypatch.setattr(ai, 'infer_slice', infer)
    candidate, reports = mr.refine_mr_many(s, [ident])
    np.testing.assert_array_equal(candidate[3] == ident, proposal)
    assert not candidate[3, 18:21, 20:23].any()
    assert reports[0]['accepted_slices'] == 1 and reports[0]['removed_voxels'] == 40


@pytest.mark.parametrize('variant,flag', [
    ('shrink', 'area_ratio_outside_0.5_to_2'),
    ('grow', 'area_ratio_outside_0.5_to_2'),
    ('shift', 'low_overlap_with_prior'),
    ('nonfinite', 'nonfinite_model_output'),
])
def test_refinement_rejects_bad_area_overlap_or_nonfinite_masks(monkeypatch, variant, flag):
    s, ids = session()
    ident = ids['femur']
    s.labels[3, 14:24, 16:26] = ident

    def infer(image, box, device, *, window):
        p = np.zeros(image.shape, np.float32)
        if variant == 'shrink':
            p[17:21, 19:23] = .9
        elif variant == 'grow':
            p[:] = .9
        elif variant == 'shift':
            p[20:30, 16:26] = .9
        else:
            p[:] = np.nan
        return p, .99, 'cpu'

    monkeypatch.setattr(ai, 'infer_slice', infer)
    candidate, reports = mr.refine_mr_many(s, [ident])
    np.testing.assert_array_equal(candidate, s.labels)
    assert reports[0]['accepted_slices'] == 0 and reports[0]['flagged_slices'] == 1
    assert flag in reports[0]['slices'][0]['flags']


def test_batch_protects_original_owners_even_after_prior_deleted_voxels(monkeypatch):
    s, ids = session(roi=(slice(1, 6), slice(5, 35), slice(5, 38)))
    a, b = ids['femur'], ids['tibia']
    s.labels[3, 12:24, 10:20] = a
    s.labels[3, 12:24, 20:30] = b
    s.labels[3, 15, 30] = 59999
    s.labels[0, 12:24, 10:20] = a  # outside ROI, never refined
    first = True

    def infer(image, box, device, *, window):
        nonlocal first
        p = np.zeros(image.shape, np.float32)
        if first:
            p[12:24, 10:19] = .9
        else:
            p[12:24, 19:31] = .9
        first = False
        return p, .7, 'cpu'

    monkeypatch.setattr(ai, 'infer_slice', infer)
    snapshot = pickle.dumps(s)
    before = s.labels.copy()
    candidate, reports = mr.refine_mr_many(s, [a, b])
    assert pickle.dumps(s) == snapshot
    full = before.copy()
    full[s.roi] = candidate
    assert not full[3, 12:24, 19].any()  # vacated by a, still not available to b
    assert full[3, 15, 30] == 59999
    assert not np.any((before != 0) & (full != 0) & (before != full))
    np.testing.assert_array_equal(full[0], before[0])
    assert [r['accepted_slices'] for r in reports] == [1, 1]


def test_absent_batch_skips_model_and_clears_only_candidate_roi():
    s, ids = session(roi=(slice(1, 6), slice(5, 35), slice(5, 38)))
    acl = s.segment(ids['acl'])
    acl.presence, acl.presence_source = 'absent', 'user_stated'
    s.labels[3, 12:24, 14:22] = acl.id  # simulate an inconsistent imported mask
    s.labels[0, 1, 1] = acl.id
    snapshot = pickle.dumps(s)
    candidate, reports = mr.refine_mr_many(s, [acl.id, ids['pcl']])
    assert pickle.dumps(s) == snapshot
    assert not candidate.any()
    assert [r['status'] for r in reports] == ['skipped_absent', 'skipped_empty']
    for report in reports:
        assert_ui_report(report, 0, 'axial')
        assert report['device'] == 'not_run' and not report['slices']
        assert report['inference_performed'] is False
    assert reports[0]['removed_voxels'] == 96


def test_absent_original_ownership_stays_reserved_in_a_mixed_batch(monkeypatch):
    s, ids = session()
    a, b = ids['acl'], ids['tibia']
    s.segment(a).presence = 'absent'
    s.labels[3, 12:24, 18:20] = a
    s.labels[3, 12:24, 20:30] = b
    calls = []

    def infer(image, box, device, *, window):
        calls.append(box)
        p = np.zeros(image.shape, np.float32)
        p[12:24, 18:30] = .9
        return p, .8, 'cpu'

    monkeypatch.setattr(ai, 'infer_slice', infer)
    candidate, reports = mr.refine_mr_many(s, [a, b])
    assert len(calls) == 1
    assert not candidate[3, 12:24, 18:20].any()
    assert reports[0]['status'] == 'skipped_absent' and reports[1]['accepted_slices'] == 1


@pytest.mark.parametrize('ids_case', ['empty', 'unknown_after_valid'])
def test_batch_validates_all_ids_before_model(ids_case):
    s, ids = session()
    s.labels[3, 12:24, 14:26] = ids['femur']
    selected = [] if ids_case == 'empty' else [ids['femur'], 9000]
    snapshot = pickle.dumps(s)
    with pytest.raises(ValueError):
        mr.refine_mr_many(s, selected)
    assert pickle.dumps(s) == snapshot


def test_batch_exception_after_completed_label_preserves_all_input(monkeypatch):
    s, ids = session()
    s.labels[2, 12:24, 10:20] = ids['femur']
    s.labels[4, 12:24, 24:34] = ids['tibia']
    count = 0

    def infer(image, box, device, *, window):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError('cancelled second label')
        p = (s.labels[2] == ids['femur']).astype(np.float32)
        p[12] = 0
        return p, .8, 'cpu'

    monkeypatch.setattr(ai, 'infer_slice', infer)
    snapshot = pickle.dumps(s)
    with pytest.raises(RuntimeError, match='second label'):
        mr.refine_mr_many(s, [ids['femur'], ids['tibia']])
    assert count == 2 and pickle.dumps(s) == snapshot


def test_batch_deduplicates_ids_and_never_touches_review_or_presence(monkeypatch):
    s, ids = session()
    ident = ids['pcl']
    segment = s.segment(ident)
    segment.reviewed = True
    segment.recognition = {'manual': True}
    segment.presence, segment.presence_source = 'present', 'user_confirmed'
    s.labels[3, 12:24, 14:26] = ident
    calls = []

    def infer(image, box, device, *, window):
        calls.append(box)
        return (s.labels[3] == ident).astype(np.float32), .8, 'cpu'

    monkeypatch.setattr(ai, 'infer_slice', infer)
    snapshot = pickle.dumps(s)
    candidate, reports = mr.refine_mr_many(s, [ident, ident])
    assert pickle.dumps(s) == snapshot
    assert len(reports) == len(calls) == 1 and reports[0]['reviewed'] is False
    np.testing.assert_array_equal(candidate, s.labels)


def test_mri_paths_use_signal_alias_without_accessing_hu_attribute(monkeypatch):
    s, ids = session()
    original_volume = s.volume

    class SignalOnlyVolume:
        modality = 'MR'
        values = original_volume.values
        spacing_zyx = original_volume.spacing_zyx

        @staticmethod
        def world(points):
            return original_volume.world(points)

        @property
        def hu(self):
            pytest.fail('MRI helpers must use the signal values alias')

    s.volume = SignalOnlyVolume()
    monkeypatch.setattr(ai, 'infer_slice', central_infer)
    candidate, report = mr.predict_mr_prompt(s, ids['acl'], 'axial', (3, 10, 12), (3, 28, 32))
    assert candidate.any() and report['intensity_units'] == 'MR signal'
    s.labels[3, 12:24, 14:26] = ids['femur']

    def infer(image, box, device, *, window):
        return (s.labels[3] == ids['femur']).astype(np.float32), .8, 'cpu'

    monkeypatch.setattr(ai, 'infer_slice', infer)
    _, reports = mr.refine_mr_many(s, [ids['femur']])
    assert reports[0]['accepted_slices'] == 1
