"""Synthetic DICOM regressions for native MR geometry and scalar preservation."""
import hashlib
import json
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import pytest

from joint2stl.dicom_io import (
    MR_UID, Series, load_series, recommended_series, scan_directory, study_recommendations,
)
from test_pipeline import write_ct


SAGITTAL = (0, 1, 0, 0, 0, -1)
AXIAL = (1, 0, 0, 0, 1, 0)
CORONAL = (1, 0, 0, 0, 0, -1)
ORIGIN = np.array([-31.25, 17.5, 245.75])


def write_mr(path, uid, study, position, *, orientation=SAGITTAL, pixels=None,
             pixel_spacing=(.7, .45), thickness=3, echo_time=44, repetition_time=3010,
             slope=None, intercept=None, image_type=('ORIGINAL', 'PRIMARY', 'M', 'ND'),
             **tags):
    """Write only a temporary synthetic source; all series have anonymous labels."""
    if pixels is None:
        pixels = np.arange(48, dtype=np.uint16).reshape(6, 8)
    pixels = np.asarray(pixels)
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MR_UID
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
    ds.SOPClassUID = MR_UID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = uid
    ds.StudyInstanceUID = study
    ds.SeriesNumber = 0
    ds.SeriesDescription = 'Anonymous'
    ds.Modality = 'MR'
    ds.ImageType = list(image_type)
    ds.Rows, ds.Columns = pixels.shape
    ds.PixelSpacing = list(pixel_spacing)
    ds.ImageOrientationPatient = list(orientation)
    ds.ImagePositionPatient = list(position)
    ds.SliceThickness = thickness
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = ds.BitsStored = pixels.dtype.itemsize * 8
    ds.HighBit = ds.BitsStored - 1
    ds.PixelRepresentation = int(pixels.dtype.kind == 'i')
    if echo_time is not None:
        ds.EchoTime = echo_time
    if repetition_time is not None:
        ds.RepetitionTime = repetition_time
    if slope is not None:
        ds.RescaleSlope = slope
    if intercept is not None:
        ds.RescaleIntercept = intercept
    for key, value in tags.items():
        setattr(ds, key, value)
    ds.PixelData = pixels.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)
    return ds


def write_stack(directory, *, uid=None, study=None, count=3, spacing=3.9,
                orientation=SAGITTAL, **tags):
    uid, study = uid or generate_uid(), study or generate_uid()
    normal = np.cross(orientation[:3], orientation[3:])
    for z in range(count):
        write_mr(directory / f'{count-z}.dcm', uid, study, ORIGIN + z * spacing * normal,
                 orientation=orientation, **tags)
    return uid, study


def one_series(directory):
    series, unreadable = scan_directory(directory)
    assert unreadable == 0 and len(series) == 1
    return series[0]


def rotated_sagittal():
    c, s = np.cos(.17), np.sin(.17)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    return tuple(np.concatenate([rotation @ SAGITTAL[:3], rotation @ SAGITTAL[3:]]))


@pytest.mark.parametrize('orientation,plane', [
    (SAGITTAL, 'sagittal'), (AXIAL, 'axial'), (CORONAL, 'coronal'),
    (rotated_sagittal(), 'sagittal'),
    ((1, 0, 0, 0, -1, 0), 'axial'),
    ((2**-.5, 2**-.5, 0, 0, 0, -1), 'oblique'),
])
def test_mr_conserves_every_native_sample_and_lps_coordinate(tmp_path, orientation, plane):
    uid, study = generate_uid(), generate_uid()
    a, b = np.asarray(orientation[:3]), np.asarray(orientation[3:])
    normal = np.cross(a, b)
    raw = np.arange(4 * 6 * 8, dtype=np.uint16).reshape(4, 6, 8) + 100
    # Neither filenames nor InstanceNumber give the spatial slice order.
    for z, filename in enumerate(('c', 'a', 'd', 'b')):
        write_mr(tmp_path / f'{filename}.dcm', uid, study, ORIGIN + z * 3.9 * normal,
                 orientation=orientation, pixels=raw[z], InstanceNumber=10-z)
    series = one_series(tmp_path)
    assert series.usable, series.reason
    assert series.modality == 'MR' and series.acquisition_plane == plane
    volume = load_series(series)
    assert volume.modality == 'MR' and volume.is_mr
    assert volume.hu.shape == raw.shape  # No isotropic or anatomical resampling.
    np.testing.assert_array_equal(volume.hu, raw)
    affine = np.eye(4)
    affine[:3, :3] = np.column_stack([a * .45, b * .7, normal * 3.9])
    affine[:3, 3] = ORIGIN
    np.testing.assert_allclose(volume.affine_lps, affine, rtol=0, atol=1e-12)
    points = np.moveaxis(np.indices(raw.shape), 0, -1)
    expected = (ORIGIN + points[..., 0, None] * normal * 3.9
                + points[..., 1, None] * b * .7 + points[..., 2, None] * a * .45)
    np.testing.assert_allclose(volume.world(points), expected, rtol=0, atol=1e-12)
    assert volume.voxel_mm3 == pytest.approx(.45 * .7 * 3.9)
    np.testing.assert_allclose(volume.spacing_zyx, [3.9, .7, .45])
    assert volume.series_key == hashlib.sha256(f'{study}/{uid}'.encode()).hexdigest()[:16]


@pytest.mark.parametrize('dtype,offset', [(np.int16, -30000), (np.uint16, 65400), (np.uint32, 2**24 + 1)])
def test_mr_native_scalars_do_not_clip_wrap_or_lose_integer_precision(tmp_path, dtype, offset):
    pixels = (np.arange(48, dtype=np.int64) + offset).astype(dtype).reshape(6, 8)
    write_stack(tmp_path, pixels=pixels)
    volume = load_series(one_series(tmp_path))
    np.testing.assert_array_equal(volume.hu, np.broadcast_to(pixels, (3, 6, 8)))


def test_mr_linear_rescale_is_per_slice_and_each_missing_tag_defaults_independently(tmp_path):
    uid, study = generate_uid(), generate_uid()
    pixels = np.arange(48, dtype=np.uint16).reshape(6, 8) + 3
    conversions = [(None, None), (.5, -2), (None, 10), (2, None), (-.25, 100), (.1, .2)]
    expected = []
    for z, (slope, intercept) in enumerate(conversions):
        write_mr(tmp_path / f'{9-z}.dcm', uid, study, ORIGIN + [-3.9*z, 0, 0],
                 pixels=pixels, slope=slope, intercept=intercept, RescaleType='US')
        expected.append(pixels.astype(float) * (1 if slope is None else slope)
                        + (0 if intercept is None else intercept))
    series = one_series(tmp_path)
    assert series.usable, series.reason
    volume = load_series(series)
    np.testing.assert_array_equal(volume.hu, np.asarray(expected))


def test_mr_padding_uses_stored_range_and_zero_not_ct_air(tmp_path):
    pixels = np.arange(48, dtype=np.uint16).reshape(6, 8)
    write_stack(tmp_path, pixels=pixels, slope=2, intercept=-10,
                PixelPaddingValue=4, PixelPaddingRangeLimit=2)
    series = one_series(tmp_path)
    assert series.usable, series.reason
    volume = load_series(series)
    expected = pixels.astype(float) * 2 - 10
    expected[(pixels >= 2) & (pixels <= 4)] = 0
    np.testing.assert_array_equal(volume.hu, np.broadcast_to(expected, volume.hu.shape))
    assert volume.hu[0, 0, 0] == -10 and volume.hu[0, 0, 6] == 2


@pytest.mark.parametrize('defect', [
    'duplicate_position', 'duplicate_instance', 'irregular_gap', 'small_irregular_gap',
    'mixed_direction', 'in_plane_drift', 'matrix', 'pixel_spacing', 'localizer',
    'scout', 'derived', 'multiframe', 'enhanced_sop', 'color', 'zero_slope', 'nonlinear_lut',
])
def test_mr_rejects_nonstackable_or_unsupported_input(tmp_path, defect):
    write_stack(tmp_path)
    path = tmp_path / '2.dcm'
    ds = pydicom.dcmread(path)
    if defect == 'duplicate_position':
        ds.ImagePositionPatient = list(ORIGIN)
    elif defect == 'duplicate_instance':
        ds.SOPInstanceUID = pydicom.dcmread(tmp_path / '3.dcm').SOPInstanceUID
    elif defect in ('irregular_gap', 'small_irregular_gap'):
        ds.ImagePositionPatient = list(ORIGIN + [-5 if defect == 'irregular_gap' else -3.92, 0, 0])
    elif defect == 'mixed_direction':
        ds.ImageOrientationPatient = list(CORONAL)
    elif defect == 'in_plane_drift':
        ds.ImagePositionPatient = list(np.asarray(ds.ImagePositionPatient) + [0, .2, 0])
    elif defect == 'matrix':
        ds.Rows += 1
    elif defect == 'pixel_spacing':
        ds.PixelSpacing = [1, 1]
    elif defect in ('localizer', 'scout'):
        ds.ImageType = ['ORIGINAL', 'PRIMARY', defect.upper()]
    elif defect == 'derived':
        ds.ImageType = ['DERIVED', 'SECONDARY', 'M']
    elif defect == 'multiframe':
        ds.NumberOfFrames = 2
    elif defect == 'enhanced_sop':
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4.1'
    elif defect == 'color':
        ds.SamplesPerPixel = 3
    elif defect == 'zero_slope':
        ds.RescaleSlope = 0
    elif defect == 'nonlinear_lut':
        lut = Dataset()
        lut.LUTDescriptor = [2, 0, 16]
        lut.LUTData = b'\0\0\1\0'
        ds.ModalityLUTSequence = [lut]
    ds.save_as(path, enforce_file_format=True)
    series = one_series(tmp_path)
    assert not series.usable and series.reason
    assert recommended_series([series]) is None
    with pytest.raises(ValueError):
        load_series(series)


@pytest.mark.parametrize('count', [1, 2])
def test_mr_single_layers_and_pairs_do_not_make_a_volume(tmp_path, count):
    write_stack(tmp_path, count=count)
    assert not one_series(tmp_path).usable


def test_anonymous_nine_image_mixed_plane_scout_is_rejected_without_description_hint(tmp_path):
    uid, study = generate_uid(), generate_uid()
    for index, orientation in enumerate((AXIAL, CORONAL, SAGITTAL) * 3):
        write_mr(tmp_path / f'{index}.dcm', uid, study, ORIGIN + [0, 0, index],
                 orientation=orientation, echo_time=3.69, repetition_time=7.8)
    series = one_series(tmp_path)
    assert len(series.files) == 9 and not series.usable and not series.recommended


def test_anonymous_long_tr_sagittal_is_preferred_and_short_tr_t1_can_be_loaded(tmp_path):
    study = generate_uid()
    for name, orientation, te, tr, thickness in (
        ('t1', SAGITTAL, 8, 500, 1), ('candidate', rotated_sagittal(), 62, 2449.5107421875, 3),
        ('axial', AXIAL, 41, 5000, 1), ('coronal', CORONAL, 25, 4000, 1),
    ):
        write_stack(tmp_path / name, study=study, orientation=orientation, echo_time=te,
                    repetition_time=tr, thickness=thickness)
    series, errors = scan_directory(tmp_path)
    assert errors == 0 and all(s.usable for s in series)
    assert {s.description for s in series} == {'Anonymous'} and {s.number for s in series} == {'0'}
    preferred = recommended_series(series)
    assert preferred is series[0] and preferred.recommended
    assert preferred.echo_time_ms == 62 and preferred.repetition_time_ms == 2449.5107421875
    assert preferred.acquisition_plane == 'sagittal' and 'TE 62 ms' in preferred.mr_recommendation
    assert preferred.reconstruction_kind == 'mr'
    manual = next(s for s in series if s.repetition_time_ms == 500)
    assert not manual.recommended and manual.usable
    assert load_series(manual).is_mr
    assert study_recommendations(series)[preferred.study_key] is preferred


@pytest.mark.parametrize('echo_time,repetition_time', [(None, 3000), (44, None), (44, 500), (0, 3000)])
def test_missing_or_short_acquisition_timing_requires_manual_choice(tmp_path, echo_time, repetition_time):
    write_stack(tmp_path, echo_time=echo_time, repetition_time=repetition_time)
    series = one_series(tmp_path)
    assert series.usable, series.reason
    assert recommended_series([series]) is None
    assert load_series(series).is_mr


@pytest.mark.parametrize('tag', ['EchoTime', 'RepetitionTime'])
def test_inconsistent_timing_is_not_reported_as_a_single_measured_value(tmp_path, tag):
    write_stack(tmp_path)
    path = tmp_path / '2.dcm'
    ds = pydicom.dcmread(path)
    setattr(ds, tag, float(getattr(ds, tag)) + 10)
    ds.save_as(path, enforce_file_format=True)
    series = one_series(tmp_path)
    assert series.usable and recommended_series([series]) is None
    metadata_key = 'echo_time_ms' if tag == 'EchoTime' else 'repetition_time_ms'
    assert load_series(series).reconstruction[metadata_key] is None


def test_multiple_mr_studies_have_individual_recommendations_and_no_default(tmp_path):
    for index in range(2):
        study = generate_uid()
        write_stack(tmp_path / str(index) / 'long', study=study)
        write_stack(tmp_path / str(index) / 'short', study=study, echo_time=8, repetition_time=500)
    series, _ = scan_directory(tmp_path)
    assert recommended_series(series) is None and not any(s.recommended for s in series)
    per_study = study_recommendations(series)
    assert len(per_study) == 2
    assert all(s is not None and s.modality == 'MR' and s.repetition_time_ms == 3010 for s in per_study.values())


@pytest.mark.parametrize('same_study,kernel', [(True, 'STANDARD'), (True, 'BONE'), (False, 'STANDARD')])
def test_mixed_ct_mr_has_no_implicit_modality_or_cross_study_choice(tmp_path, same_study, kernel):
    study = generate_uid()
    write_stack(tmp_path / 'mr', study=study)
    ct_study, uid = (study if same_study else generate_uid()), generate_uid()
    for z in range(3):
        write_ct(tmp_path / f'ct{z}.dcm', uid, ct_study, z, kernel=kernel)
    series, _ = scan_directory(tmp_path)
    assert all(s.usable for s in series) and {s.modality for s in series} == {'CT', 'MR'}
    assert recommended_series(series) is None and not any(s.recommended for s in series)
    per_study = study_recommendations(series)
    assert len(per_study) == (1 if same_study else 2)
    if same_study:
        assert all(s is None for s in per_study.values())
    else:
        assert all(s is not None for s in per_study.values())


@pytest.mark.parametrize('tags,expected', [
    ({}, 'unknown'), ({'ImageLaterality': 'L'}, 'L'), ({'Laterality': 'R'}, 'R'),
    ({'ImageLaterality': 'L', 'Laterality': 'L'}, 'L'),
    ({'ImageLaterality': 'L', 'Laterality': 'R'}, 'unknown'),
    ({'ImageLaterality': 'B'}, 'unknown'), ({'Laterality': 'U'}, 'unknown'),
])
def test_laterality_is_only_allowlisted_l_r_or_unknown(tmp_path, tags, expected):
    write_stack(tmp_path, **tags)
    series = one_series(tmp_path)
    assert series.laterality == expected
    assert load_series(series).reconstruction['laterality'] == expected


def test_conflicting_slice_laterality_stays_unknown(tmp_path):
    write_stack(tmp_path, ImageLaterality='L')
    path = tmp_path / '2.dcm'
    ds = pydicom.dcmread(path)
    ds.ImageLaterality = 'R'
    ds.save_as(path, enforce_file_format=True)
    assert load_series(one_series(tmp_path)).reconstruction['laterality'] == 'unknown'


def test_mr_acquisition_metadata_is_allowlisted_and_has_no_clinical_uids(tmp_path, monkeypatch):
    uid, study = write_stack(
        tmp_path, ScanningSequence=['SE', 'IR'], SequenceVariant='SK', ScanOptions='FS',
        MRAcquisitionType='2D', EchoTrainLength=8, FlipAngle=150,
        MagneticFieldStrength=3, SpacingBetweenSlices=3.9, ImageLaterality='R',
        PatientName='SYNTHETIC^DO_NOT_KEEP', PatientID='SYNTHETIC_PRIVATE_ID',
        AccessionNumber='SYNTHETIC_ORDER', InstitutionName='SYNTHETIC_INSTITUTION',
    )
    original_read = pydicom.dcmread
    read_tags = []

    def record_read(*args, **kwargs):
        read_tags.append(set(kwargs['specific_tags']))
        return original_read(*args, **kwargs)

    monkeypatch.setattr(pydicom, 'dcmread', record_read)
    series = one_series(tmp_path)
    volume = load_series(series)
    meta = volume.reconstruction
    assert meta['echo_time_ms'] == 44 and meta['repetition_time_ms'] == 3010
    assert meta['acquisition_plane'] == 'sagittal' and meta['laterality'] == 'R'
    assert meta['scanning_sequence'] == 'SE / IR' and meta['scan_options'] == 'FS'
    assert meta['sequence_variant'] == 'SK' and meta['mr_acquisition_type'] == '2D'
    assert meta['echo_train_length'] == 8 and meta['flip_angle_deg'] == 150
    assert meta['magnetic_field_strength_t'] == 3 and meta['spacing_between_slices_mm'] == 3.9
    np.testing.assert_array_equal(meta['scanned_axes_lps']['x'], SAGITTAL[:3])
    np.testing.assert_array_equal(meta['scanned_axes_lps']['y'], SAGITTAL[3:])
    np.testing.assert_array_equal(meta['scanned_axes_lps']['z'], [-1, 0, 0])
    forbidden = {'PatientName', 'PatientID', 'AccessionNumber', 'InstitutionName', 'PatientBirthDate'}
    assert all(not tags & forbidden for tags in read_tags)
    encoded = json.dumps(meta)
    assert uid not in encoded and study not in encoded and 'SYNTHETIC' not in encoded
    assert not any('UID' in key.upper() for key in meta)


def test_ct_defaults_and_strict_axial_hu_checks_are_retained(tmp_path):
    assert Series('fixture', '0', 'Anonymous').modality == 'CT'
    study = generate_uid()
    for name, kwargs in (
        ('valid', {'kernel': 'STANDARD'}), ('sagittal', {'orientation': SAGITTAL}),
        ('non_axial_type', {'image_type': ('ORIGINAL', 'PRIMARY', 'M')}),
    ):
        uid = generate_uid()
        for z in range(3):
            write_ct(tmp_path / f'{name}{z}.dcm', uid, study, z, **kwargs)
    series, _ = scan_directory(tmp_path)
    assert sum(s.usable for s in series) == 1
    preferred = recommended_series(series)
    assert preferred.modality == 'CT' and preferred.kernel == 'STANDARD'
    volume = load_series(preferred)
    assert volume.modality == 'CT' and not volume.is_mr
    np.testing.assert_array_equal(volume.hu, np.full((3, 16, 20), -923))


def test_ct_still_requires_both_hu_rescale_parameters(tmp_path):
    uid, study = generate_uid(), generate_uid()
    for z in range(3):
        path = tmp_path / f'{z}.dcm'
        write_ct(path, uid, study, z, kernel='STANDARD')
        ds = pydicom.dcmread(path)
        del ds.RescaleSlope
        del ds.RescaleIntercept
        ds.save_as(path, enforce_file_format=True)
    series = one_series(tmp_path)
    assert not series.usable and 'HU' in series.reason
