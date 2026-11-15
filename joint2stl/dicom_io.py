"""Classic CT/MR import on the native LPS grid, using allowlisted metadata."""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom

from .model import Volume

CT_UID = '1.2.840.10008.5.1.4.1.1.2'
MR_UID = '1.2.840.10008.5.1.4.1.1.4'
TAGS = ['SOPClassUID', 'SOPInstanceUID', 'StudyInstanceUID', 'SeriesInstanceUID',
        'SeriesNumber', 'SeriesDescription', 'Modality', 'ImageType', 'Rows', 'Columns',
        'PixelSpacing', 'ImagePositionPatient', 'ImageOrientationPatient', 'NumberOfFrames',
        'SliceThickness', 'SamplesPerPixel', 'RescaleSlope', 'RescaleIntercept',
        'RescaleType', 'ModalityLUTSequence', 'ConvolutionKernel', 'ConvolutionKernelGroup', 'Manufacturer',
        'EchoTime', 'RepetitionTime', 'ScanningSequence', 'SequenceVariant', 'ScanOptions',
        'MRAcquisitionType', 'EchoTrainLength', 'FlipAngle', 'MagneticFieldStrength',
        'SpacingBetweenSlices', 'ImageLaterality', 'Laterality']
PIXEL_TAGS=['SOPClassUID','Rows','Columns','SamplesPerPixel','PhotometricInterpretation',
            'PlanarConfiguration','NumberOfFrames','BitsAllocated','BitsStored','HighBit',
            'PixelRepresentation','PixelData','FloatPixelData','DoubleFloatPixelData',
            'RescaleSlope','RescaleIntercept','PixelPaddingValue','PixelPaddingRangeLimit']


@dataclass
class Series:
    key: str
    number: str
    description: str
    files: list[str] = field(default_factory=list)
    shape: tuple = ()
    spacing: tuple = ()
    thickness: float | None = None
    affine: np.ndarray | None = None
    reason: str = ''
    sort_score: tuple = (999, 999)
    source_label: str = ''
    kernel: str = ''
    kernel_group: str = ''
    manufacturer: str = ''
    reconstruction_kind: str = 'unknown'
    recommendation_note: str = ''
    recommended: bool = False
    study_key: str = ''
    study_label: str = ''
    modality: str = 'CT'
    echo_time_ms: float | None = None
    repetition_time_ms: float | None = None
    acquisition_plane: str = ''
    mr_recommendation: str = ''
    acquisition_metadata: dict = field(default_factory=dict)
    laterality: str = 'unknown'

    @property
    def usable(self):
        return not self.reason and self.affine is not None

    @property
    def effective_section_mm(self):
        return max(self.spacing[2], self.thickness or self.spacing[2]) if self.spacing else float('inf')


def _tag_text(value):
    if isinstance(value, str):
        return value.strip()
    try:
        return ' / '.join(str(part).strip() for part in value)
    except TypeError:
        return str(value).strip()


def classify_reconstruction(series):
    if series.modality == 'MR':
        return 'mr', _mr_note(series)
    text = re.sub(r'[_/\-]+', ' ', f'{series.kernel} {series.kernel_group} {series.description}'.upper())
    if re.search(r'\b(?:BONE(?:PLUS)?|V?SHARP|LUNG|EDGE|DETAIL)\b', text) or any(word in text for word in ('骨核', '骨算法', '骨窗')):
        return 'sharp', '骨 / 锐利核，不作自动推荐'
    # Siemens regular/filtered kernels in the soft-to-medium range. Numeric
    # suffixes such as the extra "3" in [Br40s, 3] are not separate kernels.
    codes = [int(n) for n in re.findall(r'\b(?:BR|BF|BV|B|HR|HF|H|IR|I)(\d{2})[A-Z]?\b', text)]
    if any(n >= 50 for n in codes):
        return 'sharp', '骨 / 锐利重建核，不作自动推荐'
    if (re.search(r'\b(?:SOFT|STANDARD|STND|STD|SMOOTH)\b', text)
            or any(word in text for word in ('软组织', '软核', '平滑核'))
            or any(n <= 45 for n in codes)
            or ('PHILIPS' in series.manufacturer.upper() and series.kernel.upper() in ('A', 'B'))):
        return 'soft', '软组织重建核'
    return 'unknown', '重建核不明确，请核对后手动选择'


def _long_tr_sagittal(series):
    # A selection aid based on measured acquisition tags, not a T1/PD/T2 or
    # diagnostic classification. Short-TR sagittal data can be selected
    # manually; other acquisition planes remain importable for browsing.
    return (series.modality == 'MR' and series.acquisition_plane == 'sagittal'
            and series.repetition_time_ms is not None and series.repetition_time_ms >= 1500
            and series.echo_time_ms is not None and series.echo_time_ms > 0)


def _mr_note(series):
    plane = {'sagittal': '矢状', 'coronal': '冠状', 'axial': '轴位', 'oblique': '斜位'}.get(series.acquisition_plane, '方向待核对')
    te = f'{series.echo_time_ms:g} ms' if series.echo_time_ms is not None else '未提供或不一致'
    tr = f'{series.repetition_time_ms:g} ms' if series.repetition_time_ms is not None else '未提供或不一致'
    choice = ('长 TR 矢状采集，优先用于 MRI 建模' if _long_tr_sagittal(series)
              else '矢状采集，可手动选用（包括 T1）' if series.acquisition_plane=='sagittal'
              else '仅供浏览，三维建模请选择原始（斜）矢状位')
    return f'{plane}采集；DICOM TE {te} / TR {tr}；{choice}；需人工确认对比和结构可见性'


def series_preference_key(series):
    if series.modality == 'MR':
        rank = 0 if _long_tr_sagittal(series) else (1 if series.acquisition_plane == 'sagittal' else 2)
    else:
        thin = series.effective_section_mm <= 1.5 + 1e-3
        rank = {'soft': 0 if thin else 1, 'unknown': 2 if thin else 3, 'sharp': 4, 'mixed': 5}.get(series.reconstruction_kind, 5)
    return (not series.usable, rank, series.effective_section_mm,
            series.spacing[2] if series.spacing else float('inf'),
            series.spacing[0]*series.spacing[1] if series.spacing else float('inf'), series.number)


def recommended_series(series):
    usable = [s for s in series if s.usable]
    if len({s.study_key for s in usable}) > 1 or len({s.modality for s in usable}) > 1:
        return None
    eligible = [s for s in usable if _long_tr_sagittal(s) or
                (s.modality == 'CT' and s.reconstruction_kind == 'soft'
                 and s.thickness is not None and s.effective_section_mm <= 1.5 + 1e-3)]
    return min(eligible, key=series_preference_key) if eligible else None


def study_recommendations(series):
    groups = defaultdict(list)
    for s in series:
        groups[s.study_key].append(s)
    return {key: recommended_series(group) for key, group in groups.items()}


def _vector(ds, key, length):
    val = getattr(ds, key, None)
    if val is None or len(val) != length:
        raise ValueError(f'缺少 {key}')
    v = np.asarray(val, dtype=float)
    if not np.isfinite(v).all():
        raise ValueError(f'{key} 存在无效数值')
    return v


def _image_types(ds):
    value = getattr(ds, 'ImageType', [])
    return [str(x).strip().upper() for x in ([value] if isinstance(value, str) else value)]


def _localizer(kind):
    return any(token in {'LOCALIZER', 'SCOUT', 'SURVEY', 'TOPOGRAM'}
               for part in kind for token in re.split(r'[^A-Z]+', part))


def _consistent_number(records, key, *, allow_zero=False):
    try:
        values = np.array([float(getattr(ds, key)) for _, ds in records])
    except (AttributeError, TypeError, ValueError):
        return None
    if not np.isfinite(values).all() or np.any(values < 0 if allow_zero else values <= 0):
        return None
    return float(values[0]) if np.allclose(values, values[0], rtol=1e-5, atol=1e-3) else None


def _laterality(records):
    # Never derive patient side from an image centre, normal, or filename.
    values = {str(getattr(ds, key)).strip().upper() for _, ds in records
              for key in ('ImageLaterality', 'Laterality') if getattr(ds, key, None)}
    return next(iter(values)) if len(values) == 1 and values <= {'L', 'R'} else 'unknown'


def _mr_metadata(records):
    metadata = {}
    for source, target in (('ScanningSequence', 'scanning_sequence'), ('SequenceVariant', 'sequence_variant'),
                           ('ScanOptions', 'scan_options'), ('MRAcquisitionType', 'mr_acquisition_type')):
        values = {_tag_text(getattr(ds, source, '')) for _, ds in records}
        metadata[target] = next(iter(values)) if len(values) == 1 else ''
    for source, target in (('EchoTrainLength', 'echo_train_length'), ('FlipAngle', 'flip_angle_deg'),
                           ('MagneticFieldStrength', 'magnetic_field_strength_t'),
                           ('SpacingBetweenSlices', 'spacing_between_slices_mm')):
        metadata[target] = _consistent_number(records, source, allow_zero=source == 'FlipAngle')
    return metadata


def _inspect(records, key):
    ds = records[0][1]
    s = Series(key, str(getattr(ds, 'SeriesNumber', '')), str(getattr(ds, 'SeriesDescription', '未描述序列')),
               files=[r[0] for r in records], modality=str(getattr(ds, 'Modality', '')))
    s.laterality = _laterality(records)
    if s.modality == 'MR':
        s.echo_time_ms = _consistent_number(records, 'EchoTime', allow_zero=True)
        s.repetition_time_ms = _consistent_number(records, 'RepetitionTime')
        s.acquisition_metadata = _mr_metadata(records)
    kernels = {_tag_text(getattr(d, 'ConvolutionKernel', '')) for _, d in records}
    kernels.discard('')
    s.kernel = ' | '.join(sorted(kernels))
    s.kernel_group = _tag_text(getattr(ds, 'ConvolutionKernelGroup', ''))
    s.manufacturer = str(getattr(ds, 'Manufacturer', ''))
    s.reconstruction_kind, s.recommendation_note = classify_reconstruction(s)
    if s.modality == 'CT' and len(kernels) > 1:
        s.reconstruction_kind, s.recommendation_note = 'mixed', '序列内重建核不一致，请人工核对'
    try:
        mr = s.modality == 'MR'
        if s.modality not in ('CT', 'MR'):
            raise ValueError('仅支持普通单帧 CT / MR')
        if any(str(getattr(d, 'SOPClassUID', '')) != (MR_UID if mr else CT_UID) for _, d in records):
            raise ValueError(f'仅支持普通单帧 {s.modality}；截图 / 报告 / 增强多帧不参与重建')
        image_type = _image_types(ds)
        if mr:
            if _localizer(image_type) or 'ORIGINAL' not in image_type or 'DERIVED' in image_type:
                raise ValueError('定位片或派生图像，请选择原始 MR 采集')
        elif 'LOCALIZER' in image_type or 'AXIAL' not in image_type or 'ORIGINAL' not in image_type:
            raise ValueError('定位片或派生图像，请选择原始轴位 CT')
        if len(records) < 3:
            raise ValueError('切片不足，不能组成三维体数据')
        rows, cols = int(ds.Rows), int(ds.Columns)
        ps = _vector(ds, 'PixelSpacing', 2)
        iop = _vector(ds, 'ImageOrientationPatient', 6)
        if min(ps) <= 0 or min(rows, cols) < 2:
            raise ValueError('像素尺寸无效')
        a, b = iop[:3], iop[3:]
        if not (np.isclose(np.linalg.norm(a), 1, atol=1e-4) and np.isclose(np.linalg.norm(b), 1, atol=1e-4) and abs(a @ b) < 1e-4):
            raise ValueError('图像方向不正交')
        normal = np.cross(a, b)
        if not mr and abs(normal[2]) < 0.9:
            raise ValueError('当前版本请选择原始轴位 CT，而非冠状或矢状重建')
        positioned = []
        seen = set()
        for path, d in records:
            kind = _image_types(d)
            if mr and (str(getattr(d, 'Modality', '')) != 'MR' or 'ORIGINAL' not in kind
                       or 'DERIVED' in kind or _localizer(kind)):
                raise ValueError('序列内混有定位片或非原始 MR 图像')
            if not mr and (str(getattr(d,'Modality',''))!='CT' or 'ORIGINAL' not in kind or 'AXIAL' not in kind or 'LOCALIZER' in kind):
                raise ValueError('序列内混有非原始轴位 CT 图像')
            if getattr(d,'ModalityLUTSequence',None):
                if mr:
                    raise ValueError('当前版本不支持非线性 MR 模态 LUT')
                raise ValueError('当前版本不支持非线性模态 LUT，请选择线性 HU 序列')
            if not mr and str(getattr(d,'RescaleType','')).upper() not in ('','HU'):
                raise ValueError('像素单位不是 HU，不能使用骨密度阈值')
            if not mr and ('RescaleSlope' not in d or 'RescaleIntercept' not in d):
                raise ValueError('缺少线性 HU 转换参数')
            slope, intercept = float(getattr(d, 'RescaleSlope', 1)), float(getattr(d, 'RescaleIntercept', 0))
            if not np.isfinite([slope, intercept]).all() or slope == 0:
                raise ValueError('MR 线性转换参数无效' if mr else 'HU 转换参数无效')
            if int(getattr(d, 'NumberOfFrames', 1) or 1) != 1 or int(getattr(d, 'SamplesPerPixel', 1)) != 1:
                raise ValueError(f'暂不支持多帧或彩色 {s.modality}')
            if (int(d.Rows), int(d.Columns)) != (rows, cols) or not np.allclose(_vector(d, 'PixelSpacing', 2), ps, atol=1e-5):
                raise ValueError('序列内矩阵或像素尺寸不一致')
            if not np.allclose(_vector(d, 'ImageOrientationPatient', 6), iop, atol=1e-5):
                raise ValueError('序列内方向变化，不能直接堆叠')
            uid = str(getattr(d, 'SOPInstanceUID', path))
            if uid in seen:
                raise ValueError('存在重复实例，请整理序列后导入')
            seen.add(uid)
            p = _vector(d, 'ImagePositionPatient', 3)
            positioned.append((float(p @ normal), path, p))
        positioned.sort(key=lambda x: x[0])
        coords = np.array([r[0] for r in positioned])
        deltas = np.diff(coords)
        dz = float(np.median(deltas))
        if dz <= 1e-4 or np.any(deltas < 1e-4):
            raise ValueError('重复层位置，无法确定层间距')
        if np.max(np.abs(deltas - dz)) > (0.01 if mr else max(0.01, dz * 0.02)):
            raise ValueError('存在缺层或不规则间距，当前版本不自动插补')
        origin = positioned[0][2]
        positions = np.array([r[2] for r in positioned])
        expected = origin + (coords - coords[0])[:, None] * normal
        if mr:
            expected = origin + (coords - coords[0])[:, None] * normal / (normal @ normal)
        if np.max(np.linalg.norm(positions - expected, axis=1)) > 0.05:
            raise ValueError('存在机架倾斜或平面内漂移，需先校正')
        affine = np.eye(4)
        # Use the measured MR slice displacement, preserving the native sample
        # grid and sub-micron DICOM rounding without normalising/reorienting it.
        step = (positions[-1] - origin) / (len(records) - 1) if mr else normal * dz
        if mr and np.max(np.linalg.norm(positions - (origin + np.arange(len(records))[:, None] * step), axis=1)) > 0.01:
            raise ValueError('MR 层位置不在规则原始网格内，当前版本不自动插补')
        affine[:3, :3] = np.column_stack([a * ps[1], b * ps[0], step])
        affine[:3, 3] = origin
        s.files = [r[1] for r in positioned]
        s.affine, s.shape = affine, (len(records), rows, cols)
        s.spacing = (float(ps[1]), float(ps[0]), float(np.linalg.norm(step)) if mr else dz)
        if mr:
            unit_normal = normal / np.linalg.norm(normal)
            axis = int(np.argmax(np.abs(unit_normal)))
            s.acquisition_plane = ('sagittal', 'coronal', 'axial')[axis] if abs(unit_normal[axis]) >= 0.9 else 'oblique'
            s.acquisition_metadata.update(
                echo_time_ms=s.echo_time_ms, repetition_time_ms=s.repetition_time_ms,
                acquisition_plane=s.acquisition_plane,
                scanned_axes_lps={name: (affine[:3, i] / np.linalg.norm(affine[:3, i])).tolist()
                                  for i, name in enumerate(('x', 'y', 'z'))})
        thickness = getattr(ds, 'SliceThickness', None)
        s.thickness = float(thickness) if thickness is not None else None
        if s.thickness is not None and (not np.isfinite(s.thickness) or s.thickness <= 0):
            s.thickness = None
        s.sort_score = (dz, ps[0] * ps[1])
    except (ValueError, TypeError, AttributeError) as exc:
        s.reason = str(exc)
    if s.modality == 'MR':
        s.reconstruction_kind, s.mr_recommendation = classify_reconstruction(s)
        s.recommendation_note = s.mr_recommendation
    return s


def scan_directory(directory, progress=lambda *_: None):
    root = Path(directory)
    if not root.is_dir():
        raise ValueError('请选择包含 DICOM 的文件夹。')
    # Extensionless DICOM is common on exported media. Ignore known unrelated files.
    paths = sorted(p for p in root.rglob('*') if p.is_file() and not p.name.startswith('.')
                   and (p.suffix.lower() in {'.dcm', '.dicom', '.ima', ''}))
    if not paths:
        raise ValueError('此文件夹没有找到 DICOM 文件。')
    groups = defaultdict(list)
    unreadable = 0
    for i, path in enumerate(paths):
        if i % 32 == 0:
            progress(int(i / len(paths) * 95), f'检查序列 {i + 1} / {len(paths)}')
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=TAGS)
            if 'SeriesInstanceUID' not in ds or 'StudyInstanceUID' not in ds:
                unreadable += 1
                continue
            uid = str(ds.StudyInstanceUID) + '/' + str(ds.SeriesInstanceUID)
            groups[uid].append((str(path), ds))
        except (OSError, pydicom.errors.InvalidDicomError, ValueError):
            unreadable += 1
    result = []
    for uid, records in groups.items():
        series = _inspect(records, hashlib.sha256(uid.encode()).hexdigest()[:16])
        series.study_key = hashlib.sha256(str(records[0][1].StudyInstanceUID).encode()).hexdigest()[:16]
        result.append(series)
    for series in result:
        parent=Path(series.files[0]).parent
        relative=parent.relative_to(root)
        series.source_label=root.name if str(relative)=='.' else relative.as_posix()
    result.sort(key=series_preference_key)
    study_numbers = {}
    for series in result:
        series.study_label = f'检查 {study_numbers.setdefault(series.study_key, len(study_numbers) + 1):02d}'
    preferred = recommended_series(result)
    for series in result:
        series.recommended = series is preferred
    progress(100, f'找到 {sum(s.usable for s in result)} 套可用序列')
    return result, unreadable


def load_series(series: Series, progress=lambda *_: None):
    if not series.usable:
        raise ValueError(series.reason or '此序列不可用。')
    first = pydicom.dcmread(series.files[0], stop_before_pixels=True, specific_tags=TAGS)
    slope, intercept = float(getattr(first, 'RescaleSlope', 1)), float(getattr(first, 'RescaleIntercept', 0))
    integer = ((slope == 1 and intercept == 0) if series.modality == 'MR'
               else slope == 1 and intercept == int(intercept) and -16384 <= intercept <= 0)
    data = np.empty(series.shape, dtype=np.int16 if integer else np.float32)
    for z, path in enumerate(series.files):
        if z % 16 == 0:
            progress(int(z / len(series.files) * 100), f'读取 {series.modality} {z + 1} / {len(series.files)}')
        ds = pydicom.dcmread(path,specific_tags=PIXEL_TAGS)
        raw = ds.pixel_array
        if raw.shape != series.shape[1:]:
            raise ValueError(f'第 {z + 1} 层像素矩阵异常。')
        values = raw.astype(np.float64 if series.modality == 'MR' else np.float32) * float(getattr(ds, 'RescaleSlope', 1)) + float(getattr(ds, 'RescaleIntercept', 0))
        padding=getattr(ds,'PixelPaddingValue',None)
        if padding is not None:
            other=getattr(ds,'PixelPaddingRangeLimit',None)
            other=padding if other is None else other
            lo,hi=sorted([float(padding),float(other)])
            values[(raw>=lo)&(raw<=hi)] = 0 if series.modality == 'MR' else -4096
        if not np.isfinite(values).all():
            unit = 'MR 信号' if series.modality == 'MR' else 'HU'
            raise ValueError(f'第 {z + 1} 层存在无效 {unit} 数值。')
        if data.dtype == np.int16 and (values.min() < -32768 or values.max() > 32767 or not np.equal(values, np.rint(values)).all()):
            data = data.astype(np.float32)
        if series.modality == 'MR' and data.dtype == np.float32 and not np.array_equal(values, values.astype(np.float32).astype(np.float64)):
            data = data.astype(np.float64)
        data[z] = values
    progress(100, f'{series.modality} 已读取')
    reconstruction = {'series_number': series.number, 'kernel': series.kernel,
                      'kind': series.reconstruction_kind, 'thickness_mm': series.thickness,
                      'spacing_xyz_mm': list(series.spacing), 'laterality': series.laterality}
    if series.modality == 'MR':
        reconstruction.update(series.acquisition_metadata)
        reconstruction['mr_recommendation'] = series.mr_recommendation
    return Volume(data, series.affine.copy(), series.description, str(Path(series.files[0]).parent),
                  series.key, reconstruction, modality=series.modality)
