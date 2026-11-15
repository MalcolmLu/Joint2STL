"""Geometry/outline regressions exercise failure modes, not clinical accuracy."""
import numpy as np
import pytest

from joint2stl.anatomy import identify_bones
from joint2stl.model import Session, Volume, load_project, save_project
from joint2stl.preprocessing import patient_region, remove_segmented_table, exclude_external_attachments
from joint2stl.segmentation import initial_segmentation, install_initial
from joint2stl.workflow import install_recognition, prepare_initial


def knee_fixture(bilateral=True, table=True):
    hu = np.full((100, 190, 260), -1000, np.int16)
    labels = np.zeros(hu.shape, np.uint16)
    zz, yy, xx = np.ogrid[:100, :190, :260]
    centers = [(70, '右侧'), (190, '左侧')] if bilateral else [(70, '右侧')]
    expected = {}
    for index, (x, side) in enumerate(centers):
        hu[np.broadcast_to(((xx-x)/48)**2+((yy-92)/65)**2 < 1, hu.shape)] = 35
        sign = -1 if side == '右侧' else 1
        # IDs and volumes deliberately do not encode anatomical ordering.
        bones = [(7+index*10, '股骨', (76, 91, x), (23, 19, 20)),
                 (2+index*10, '胫骨', (25, 94, x+2), (23, 18, 17)),
                 (9+index*10, '髌骨', (64, 47, x-3), (14, 6, 11)),
                 (4+index*10, '腓骨', (24, 111, x+sign*29), (23, 6, 5))]
        for ident, name, center, radii in bones:
            z, y, bx = center; rz, ry, rx = radii
            mask = ((zz-z)/rz)**2+((yy-y)/ry)**2+((xx-bx)/rx)**2 < 1
            labels[mask] = ident; hu[mask] = 900; expected[ident] = side+'_'+name
    if table:
        hu[:, 174:180, :] = 750; labels[:, 174:180, :] = 31
    affine = np.eye(4); affine[:3, 3] = [-130, -90, 0]
    return Volume(hu, affine), labels, expected


@pytest.mark.parametrize('flip', [False, True])
def test_anatomy_uses_lps_not_label_id_volume_order_or_voxel_direction(flip):
    volume, labels, expected = knee_fixture()
    if flip:
        volume.hu = volume.hu[::-1, ::-1, ::-1].copy()
        labels = labels[::-1, ::-1, ::-1].copy()
        old = volume.affine_lps.copy()
        volume.affine_lps[:3, 3] = volume.world(np.array(labels.shape)-1)
        volume.affine_lps[:3, :3] = -old[:3, :3]
    proposals = identify_bones(volume, labels, 'knee', '双侧')
    names = {p['id']: p['name'] for p in proposals if p['name']}
    assert names == expected
    assert next(p for p in proposals if p['id'] == 31)['status'] == 'uncertain'


def test_naming_does_not_invent_missing_femur_from_a_tibia_fibula_pair():
    volume, labels, _ = knee_fixture(bilateral=False, table=False)
    labels[(labels != 2) & (labels != 4)] = 0
    proposals = identify_bones(volume, labels, 'knee', '右侧')
    assert all(p['name'] == '' for p in proposals)


def test_a_shared_tibia_fibula_label_is_flagged_instead_of_exported_as_confirmed_tibia():
    volume, labels, _ = knee_fixture(bilateral=False, table=False)
    labels[labels == 4] = 2
    proposals = identify_bones(volume, labels, 'knee', '右侧')
    tibia = next(p for p in proposals if p['id'] == 2)
    assert tibia['status'] == 'check_connection' and '疑似相连' in tibia['name']


def test_a_shared_femur_tibia_label_gets_a_region_hint_without_fabricating_separate_bones():
    volume, labels, _ = knee_fixture(bilateral=False, table=False)
    volume.affine_lps[2,2] = 1.2
    labels[labels == 2] = 7
    before = labels.copy()
    proposals = identify_bones(volume, labels, 'knee', '右侧')
    main = next(p for p in proposals if p['id'] == 7)
    assert main['status'] == 'check_connection' and '股骨/胫骨' in main['name']
    np.testing.assert_array_equal(labels, before)


def test_unilateral_volume_needs_side_when_no_contralateral_landmarks():
    volume, labels, _ = knee_fixture(bilateral=False, table=False)
    result = identify_bones(volume, labels, 'knee', '双侧')
    assert all(p['side'] == '侧别待定' and p['status'] == 'check_side' for p in result)


@pytest.mark.parametrize('side', ['右侧', '左侧'])
def test_shoulder_geometry_uses_lateral_humerus_posterior_scapula_and_superior_clavicle(side):
    labels = np.zeros((140, 180, 240), np.uint16)
    zz, yy, xx = np.ogrid[:140, :180, :240]
    shapes = [(3, (50, 75, 40), (46, 16, 14)), (8, (66, 116, 88), (50, 14, 33)),
              (2, (115, 58, 104), (8, 7, 42))]
    for ident, (z, y, x), (rz, ry, rx) in shapes:
        labels[((zz-z)/rz)**2+((yy-y)/ry)**2+((xx-x)/rx)**2 < 1] = ident
    if side == '左侧':
        labels = labels[:, :, ::-1].copy()
    affine = np.eye(4); affine[:3, 3] = [-120, -90, 0]
    volume = Volume(np.zeros(labels.shape, np.int16), affine)
    proposals = identify_bones(volume, labels, 'shoulder', side)
    assert {p['id']:p['name'] for p in proposals} == {3:side+'_肱骨',8:side+'_肩胛骨',2:side+'_锁骨'}


def test_patient_outline_removes_table_and_preserves_both_legs_and_cropped_roi():
    volume, labels, _ = knee_fixture()
    original = volume.hu.copy()
    keep, report = patient_region(volume, volume.full_box)
    assert not keep[labels == 31].any()
    assert keep[(labels > 0) & (labels != 31)].all()
    roi = (slice(10, 90), slice(30, 185), slice(45, 105))
    cropped, _ = patient_region(volume, roi)
    np.testing.assert_array_equal(cropped, keep[roi])
    np.testing.assert_array_equal(original, volume.hu)
    assert report['supported_slices'] == 100


def test_outline_fails_open_and_reports_uncertain_slice():
    volume, _, _ = knee_fixture(bilateral=False)
    volume.hu[30] = -1000
    volume.hu[30, 174:180] = 700
    keep, report = patient_region(volume, volume.full_box)
    assert keep[30].all() and 31 in report['preserved_uncertain_slices_1based']


@pytest.mark.parametrize('cast_hu', [230, 1000])
def test_plaster_shell_splint_and_external_brace_are_removed_with_table_without_cutting_bones(cast_hu):
    volume, labels, _ = knee_fixture(bilateral=False)
    zz, yy, xx = np.ogrid[:100, :190, :260]
    shell = (((xx-70)/56)**2+((yy-92)/73)**2 < 1) & (((xx-70)/50)**2+((yy-92)/67)**2 > 1)
    shell = np.broadcast_to(shell, labels.shape)
    volume.hu[shell] = cast_hu; labels[shell] = 21
    volume.hu[:, 65:120, 121:129] = 1600;labels[:, 65:120, 121:129] = 22
    volume.hu[:, 15:23, 50:90] = 2800;labels[:, 15:23, 50:90] = 23
    keep, _ = patient_region(volume, volume.full_box)
    filtered, report = exclude_external_attachments(labels>0, keep, volume.spacing_zyx)
    bones = (labels>0) & (labels<20)
    assert filtered[bones].all()
    assert not filtered[labels>=20].any()
    assert report['removed_external_components'] >= 2


def test_bone_connected_external_hardware_is_flagged_and_internal_bone_retained():
    volume, labels, _ = knee_fixture(bilateral=False, table=False)
    bone = labels == 2
    labels[:, 75:120, 124:145] = 25;volume.hu[:, 75:120, 124:145] = 2000
    labels[24:27, 92:97, 80:128] = 25;volume.hu[24:27, 92:97, 80:128] = 2000
    keep, _ = patient_region(volume, volume.full_box)
    filtered, report = exclude_external_attachments(labels>0, keep, volume.spacing_zyx)
    assert filtered[bone].all()
    assert report['ambiguous_attachment_components'] >= 1


def test_table_removal_candidate_is_atomic_and_undoable():
    volume, labels, _ = knee_fixture()
    session = Session(volume)
    for ident in range(1, 32):
        session.add_segment()
    session.apply(volume.full_box, labels, 'initial')
    candidate, report = remove_segmented_table(session)
    np.testing.assert_array_equal(session.labels, labels)
    assert report['removed_labeled_voxels'] == int((labels == 31).sum())
    session.apply(session.roi, candidate, '自动去床板')
    assert not (session.labels == 31).any()
    session.travel_history()
    np.testing.assert_array_equal(session.labels, labels)


def test_initial_pipeline_names_bones_and_excludes_table_before_components():
    volume, _, expected = knee_fixture()
    session = Session(volume)
    result = prepare_initial(session, session.parameters)
    assert result['segmentation'][1] == len(expected)
    assert not session.labels.any() and not session.segments
    install_initial(session, result['segmentation'], result['recognition'])
    assert {s.name for s in session.segments} == set(expected.values())
    assert all(not s.reviewed for s in session.segments)


def test_filling_cortex_does_not_turn_a_dense_skin_or_cast_ring_into_solid_leg():
    volume, labels, _ = knee_fixture(bilateral=False, table=False)
    yy, xx = np.ogrid[:190, :260]
    ring = (((xx-70)/47)**2+((yy-92)/64)**2 < 1) & (((xx-70)/46)**2+((yy-92)/63)**2 > 1)
    volume.hu[np.broadcast_to(ring, volume.hu.shape)] = 260
    local, _ = initial_segmentation(volume, volume.full_box, min_mm3=200)
    assert local[50, 90, 70] == 0  # Muscle between femur/tibia, inside the outer ring.
    assert (local[labels == 7] > 0).all() and (local[labels == 2] > 0).all()


def test_recognition_preserves_manual_name_and_undo_order_and_roundtrips(tmp_path):
    volume, labels, _ = knee_fixture(bilateral=False, table=False)
    session = Session(volume)
    for ident in range(1, 10):
        session.add_segment()
    session.apply(volume.full_box, labels, 'initial')
    session.side = '右侧'
    session.segment(7).name = '人工骨名'; session.segment(7).reviewed = True
    before = session.segment(2).name
    proposals = identify_bones(volume, labels, 'knee', '右侧')
    summary = install_recognition(session, proposals)
    assert summary['manual_preserved'] == 1
    assert session.segment(7).name == '人工骨名' and session.segment(2).name == '右侧_胫骨'
    session.travel_history()
    assert session.segment(2).name == before and session.segment(7).name == '人工骨名'
    session.travel_history(True)
    assert session.segment(2).name == '右侧_胫骨'
    np.testing.assert_array_equal(session.labels, labels)
    np.testing.assert_array_equal(session.counts(), np.bincount(labels.ravel(), minlength=10))
    session.preprocess_reports = [{'method': 'fixture'}]
    file = tmp_path/'named.j2stl'; save_project(session, file)
    restored = load_project(file)
    assert restored.segment(2).recognition == session.segment(2).recognition
    assert not restored.segment(2).reviewed
    assert restored.preprocess_reports == session.preprocess_reports
