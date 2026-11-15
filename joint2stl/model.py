from __future__ import annotations

import json
import os
import tempfile
import zlib
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

PALETTE = ['#6bb9ff', '#f8c76b', '#74d9b4', '#c39af0', '#f19a8c', '#83d3e3', '#c3d789', '#b3abff']
BONES = {'knee': ['股骨', '胫骨', '腓骨', '髌骨'], 'shoulder': ['肱骨', '肩胛骨', '锁骨'],
         'leg': ['骨盆', '股骨', '胫骨', '腓骨', '髌骨', '距骨', '跟骨', '跖骨', '趾骨']}
Box = tuple[slice, slice, slice]


def count_labels(labels,minimum=0):
    """Bound bincount's implicit int64 copy instead of copying a whole CT stack."""
    result=np.zeros(minimum,dtype=np.int64)
    flat=np.asarray(labels).reshape(-1) if labels.flags.c_contiguous else None
    if flat is None:
        chunks=(np.ascontiguousarray(plane).ravel() for plane in labels)
    else:
        chunks=(flat[i:i+2_000_000] for i in range(0,flat.size,2_000_000))
    for chunk in chunks:
        counts=np.bincount(chunk)
        if len(result)<len(counts):result=np.pad(result,(0,len(counts)-len(result)))
        result[:len(counts)]+=counts
    return result


@dataclass
class Volume:
    hu: np.ndarray
    affine_lps: np.ndarray  # maps voxel XYZ to patient LPS, in mm
    description: str = ''
    source: str = ''
    series_key: str = ''
    reconstruction: dict = field(default_factory=dict)
    modality: str = 'CT'

    def __post_init__(self):
        if self.hu.ndim != 3 or min(self.hu.shape) < 2 or self.hu.dtype.kind not in 'if':
            raise ValueError('影像必须是三维数值体数据，每个维度至少两层。')
        if self.modality not in ('CT','MR'):
            raise ValueError('仅支持 CT 或 MR 影像。')
        self.affine_lps = np.asarray(self.affine_lps, dtype=float)
        if self.affine_lps.shape != (4, 4) or not np.isfinite(self.affine_lps).all():
            raise ValueError('无效空间坐标。')
        if not np.allclose(self.affine_lps[3], [0, 0, 0, 1]):
            raise ValueError('无效空间坐标末行。')
        if abs(np.linalg.det(self.affine_lps[:3, :3])) < 1e-8:
            raise ValueError('体素尺寸无效。')

    @property
    def is_mr(self):
        return self.modality=='MR'

    @property
    def values(self):
        return self.hu

    @property
    def intensity_units(self):
        return 'MR signal' if self.is_mr else 'HU'

    @property
    def spacing_zyx(self):
        return np.linalg.norm(self.affine_lps[:3, :3], axis=0)[::-1]

    @property
    def voxel_mm3(self):
        return abs(float(np.linalg.det(self.affine_lps[:3, :3])))

    @property
    def full_box(self):
        return tuple(slice(0, n) for n in self.hu.shape)

    def world(self, zyx):
        xyz = np.asarray(zyx, dtype=float)[..., ::-1]
        return xyz @ self.affine_lps[:3, :3].T + self.affine_lps[:3, 3]


@dataclass
class Segment:
    id: int
    name: str
    color: str
    visible: bool = True
    reviewed: bool = False
    recognition: dict = field(default_factory=dict)
    kind: str = 'bone'
    structure: str = ''
    presence: str = 'unknown'
    presence_source: str = ''


@dataclass
class Seed:
    zyx: tuple[int, int, int]
    label: int


@dataclass
class Patch:
    box: Box
    shape: tuple
    before: bytes
    after: bytes
    title: str
    names_before: list | None = None
    names_after: list | None = None
    removed_segment: dict | None = None

    @property
    def nbytes(self):
        return len(self.before) + len(self.after) + len(json.dumps([self.names_before, self.names_after, self.removed_segment]).encode('utf-8'))


class History:
    def __init__(self):
        self.undo_stack: list[Patch] = []
        self.redo_stack: list[Patch] = []
        self.limit = 96 * 1024**2
        self.version=0

    def push(self, box, before, after, title, names_before=None, names_after=None, removed_segment=None):
        if np.array_equal(before, after) and names_before == names_after and removed_segment is None:
            return False
        p = Patch(box, before.shape,
                  zlib.compress(np.ascontiguousarray(before, dtype=np.uint16).tobytes(), 1),
                  zlib.compress(np.ascontiguousarray(after, dtype=np.uint16).tobytes(), 1), title,
                  names_before, names_after, removed_segment)
        self.undo_stack.append(p)
        self.redo_stack.clear()
        self.version+=1
        while len(self.undo_stack) > 1 and sum(x.nbytes for x in self.undo_stack) > self.limit:
            self.undo_stack.pop(0)
        return True

    def travel(self, labels, redo=False):
        src, dst = (self.redo_stack, self.undo_stack) if redo else (self.undo_stack, self.redo_stack)
        if not src:
            return None
        p = src.pop()
        raw = p.after if redo else p.before
        labels[p.box] = np.frombuffer(zlib.decompress(raw), dtype=np.uint16).reshape(p.shape)
        dst.append(p)
        self.version+=1
        return p.title


@dataclass
class Session:
    volume: Volume
    labels: np.ndarray | None = None
    segments: list[Segment] = field(default_factory=list)
    joint: str = 'knee'
    side: str = '双侧'
    roi: Box | None = None
    seeds: list[Seed] = field(default_factory=list)
    parameters: dict = field(default_factory=lambda: {'threshold': 200, 'upper': 4000, 'min_mm3': 200, 'fill': True, 'surface_mm': 0.35, 'compute_backend': 'auto', 'remove_table': True, 'auto_name': True, 'mesh_smoothing': 'standard', 'preserve_detail': True, 'recover_edges': True})
    ai_reports: list = field(default_factory=list)
    preprocess_reports: list = field(default_factory=list)
    repair_reports: list = field(default_factory=list)
    history: History = field(default_factory=History)
    dirty: bool = False
    revision: int = 0
    project_path: str = ''
    _counts_cache: np.ndarray | None = field(default=None,init=False,repr=False)
    _counts_version: int = field(default=0,init=False,repr=False)
    _mesh_cache: OrderedDict = field(default_factory=OrderedDict,init=False,repr=False)
    _mesh_cache_bytes: int = field(default=0,init=False,repr=False)

    def __post_init__(self):
        if self.labels is None:
            self.labels = np.zeros(self.volume.hu.shape, dtype=np.uint16)
            self._counts_cache=np.array([self.labels.size],dtype=np.int64)
        if self.labels.shape != self.volume.hu.shape or self.labels.dtype != np.uint16:
            raise ValueError('分割尺寸或类型与 CT 不一致。')
        self.roi = self.roi or self.volume.full_box
        if self.volume.is_mr:self.parameters.setdefault('mr_sagittal_only',True)
        self.set_roi(self.roi, dirty=False)

    def set_roi(self, box, dirty=True):
        if len(box) != 3 or any(s.start is None or s.stop is None or s.step not in (None, 1)
                                or not (0 <= s.start < s.stop <= n) for s, n in zip(box, self.volume.hu.shape)):
            raise ValueError('范围必须在 CT 内，起始层不能大于终止层。')
        self.roi = tuple(box)
        if dirty:
            self.dirty = True

    def next_segment_id(self):
        # A deleted ID remains reserved while its deletion can be undone/redone.
        reserved = [p.removed_segment['segment']['id']
                    for p in (*self.history.undo_stack, *self.history.redo_stack)
                    if p.removed_segment is not None]
        return max([0, *(s.id for s in self.segments), *reserved]) + 1

    def add_segment(self, name=None):
        ident = self.next_segment_id()
        if ident > 65535:
            raise ValueError('骨标签数量已达上限。')
        segment = Segment(ident, name or f'骨块 {ident:02d} · 待命名', PALETTE[(ident - 1) % len(PALETTE)])
        self.segments.append(segment)
        self.dirty = True
        return segment

    def segment(self, ident):
        return next((s for s in self.segments if s.id == ident), None)

    def remove_segment(self, ident):
        """Remove one complete label, with mask/metadata/prompts in one undo step."""
        segment = self.segment(ident)
        if segment is None:
            raise ValueError('请先选择需要删除的标签。')
        # Find a tight box one acquired layer at a time; do not allocate a
        # volume-sized coordinate array for a long CT or a large bone.
        lower = np.array(self.labels.shape)
        upper = np.zeros(3, dtype=int)
        found = False
        for z, plane in enumerate(self.labels):
            mask = plane == ident
            if not mask.any():
                continue
            ys = np.flatnonzero(mask.any(axis=1))
            xs = np.flatnonzero(mask.any(axis=0))
            lower = np.minimum(lower, [z, ys[0], xs[0]])
            upper = np.maximum(upper, [z + 1, ys[-1] + 1, xs[-1] + 1])
            found = True
        box = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper)) if found else (slice(0, 1),) * 3
        before = self.labels[box].copy()
        after = before.copy()
        after[after == ident] = 0
        removed = {
            'segment': asdict(segment), 'index': self.segments.index(segment),
            'seeds': [(i, asdict(seed)) for i, seed in enumerate(self.seeds) if seed.label == ident],
            'points': [(i, deepcopy(point)) for i, point in enumerate(self.parameters.get('mr_correction_points', []))
                       if point.get('label_id') == ident],
        }
        self.history.push(box, before, after, '删除标签：' + segment.name, removed_segment=removed)
        self.labels[box] = after
        self._restore_removed_segment(removed, redo=True)
        self._counts_cache = None
        self.changed()
        return True

    def _restore_removed_segment(self, removed, redo):
        ident = removed['segment']['id']
        if redo:
            self.segments = [s for s in self.segments if s.id != ident]
            self.seeds = [seed for seed in self.seeds if seed.label != ident]
            if 'mr_correction_points' in self.parameters:
                self.parameters['mr_correction_points'] = [
                    p for p in self.parameters['mr_correction_points'] if p.get('label_id') != ident]
        else:
            if self.segment(ident) is None:
                self.segments.insert(min(removed['index'], len(self.segments)), Segment(**deepcopy(removed['segment'])))
            for index, entry in removed['seeds']:
                seed = Seed(tuple(entry['zyx']), entry['label'])
                if seed not in self.seeds:
                    self.seeds.insert(min(index, len(self.seeds)), seed)
            if removed['points']:
                points = self.parameters.setdefault('mr_correction_points', [])
                for index, entry in removed['points']:
                    if entry not in points:
                        points.insert(min(index, len(points)), deepcopy(entry))

    def changed(self):
        self.dirty = True
        self.revision += 1
        self._mesh_cache.clear();self._mesh_cache_bytes=0

    def cached_mesh(self,key):
        result=self._mesh_cache.get(key)
        if result is not None:self._mesh_cache.move_to_end(key)
        return result

    def remember_mesh(self,key,mesh,box):
        size=mesh.vertices.nbytes+mesh.faces.nbytes
        limit=96*1024**2
        if size>limit:return
        while self._mesh_cache and self._mesh_cache_bytes+size>limit:
            _,(old,_) = self._mesh_cache.popitem(last=False)
            self._mesh_cache_bytes-=old.vertices.nbytes+old.faces.nbytes
        self._mesh_cache[key]=(mesh,box);self._mesh_cache_bytes+=size

    def apply(self, box, replacement, title):
        if replacement.shape != self.labels[box].shape:
            raise ValueError('编辑尺寸错误。')
        if self._counts_version!=self.history.version:self._counts_cache=None
        before = self.labels[box].copy()
        if self.history.push(box, before, replacement, title):
            self._adjust_counts(before,replacement)
            self._counts_version=self.history.version
            self.labels[box] = replacement
            self.changed()
            return True
        return False

    def counts(self):
        minimum=max((s.id for s in self.segments),default=0)+1
        if self._counts_cache is None or self._counts_version!=self.history.version:
            self._counts_cache=count_labels(self.labels,minimum)
            self._counts_version=self.history.version
        if len(self._counts_cache)<minimum:self._counts_cache=np.pad(self._counts_cache,(0,minimum-len(self._counts_cache)))
        return self._counts_cache

    def _adjust_counts(self,before,after):
        if self._counts_cache is None:return
        a=count_labels(before);b=count_labels(after)
        n=max(len(a),len(b),len(self._counts_cache))
        self._counts_cache=np.pad(self._counts_cache,(0,n-len(self._counts_cache)))
        self._counts_cache[:len(a)]-=a;self._counts_cache[:len(b)]+=b

    def record_preapplied(self,box,before,title):
        # Paint keeps the cache current incrementally, before history is committed.
        if self._counts_version!=self.history.version:self._counts_cache=None
        if self.history.push(box,before,self.labels[box],title):
            self._counts_version=self.history.version;self.changed();return True
        return False

    def update_bone_names(self, updates, title='自动识别骨名', preserve_reviewed=True):
        before, after = [], []
        for update in updates:
            segment = self.segment(update['id'])
            if segment is None or (preserve_reviewed and segment.reviewed):
                continue
            snapshot = asdict(segment)
            old = {key: snapshot[key] for key in ('id', 'name', 'reviewed', 'recognition')}
            new = {'id': segment.id, 'name': update['name'],
                   'reviewed': bool(update.get('reviewed', False)), 'recognition': update.get('recognition', {})}
            if old != new:
                before.append(old); after.append(new)
        if not before:
            return False
        # Metadata-only edits share the same undo order as brush and mask edits.
        box = (slice(0, 1),) * 3
        sample = self.labels[box]
        if self._counts_version != self.history.version:
            self._counts_cache = None
        self.history.push(box, sample, sample, title, before, after)
        self._counts_version = self.history.version
        self._restore_names(after)
        self.dirty = True
        return True

    def _restore_names(self, updates):
        for update in updates or []:
            segment = self.segment(update['id'])
            if segment is not None:
                for key in ('name','reviewed','recognition','kind','structure','presence','presence_source'):
                    if key in update:setattr(segment,key,update[key])

    def update_presence(self, ident, presence, source='user_confirmed'):
        segment=self.segment(ident)
        if segment is None or segment.kind!='ligament' or presence not in ('unknown','present','absent'):
            raise ValueError('请选择韧带标签和有效的存在状态。')
        before_meta=asdict(segment);after_meta={**before_meta,'presence':presence,'presence_source':source}
        occupied=np.argwhere(self.labels==ident) if presence=='absent' else np.empty((0,3),int)
        box=(tuple(slice(int(occupied[:,a].min()),int(occupied[:,a].max())+1) for a in range(3))
             if len(occupied) else (slice(0,1),)*3)
        before=self.labels[box].copy();after=before.copy()
        if presence=='absent':after[after==ident]=0
        if not self.history.push(box,before,after,'标记韧带状态',[before_meta],[after_meta]):return False
        self.labels[box]=after;self._counts_cache=None
        self._restore_names([after_meta]);self.changed();return True

    def travel_history(self,redo=False):
        if self._counts_version!=self.history.version:self._counts_cache=None
        stack=self.history.redo_stack if redo else self.history.undo_stack
        if not stack:return None
        patch=stack[-1]
        before_counts=count_labels(self.labels[patch.box]) if self._counts_cache is not None else None
        title=self.history.travel(self.labels,redo)
        if patch.removed_segment is not None:
            self._restore_removed_segment(patch.removed_segment, redo)
        self._restore_names(patch.names_after if redo else patch.names_before)
        if before_counts is not None:
            after_counts=count_labels(self.labels[patch.box])
            n=max(len(before_counts),len(after_counts),len(self._counts_cache))
            self._counts_cache=np.pad(self._counts_cache,(0,n-len(self._counts_cache)))
            self._counts_cache[:len(before_counts)]-=before_counts
            self._counts_cache[:len(after_counts)]+=after_counts
        self._counts_version=self.history.version;self.changed()
        return title


def save_project(session: Session, path: str | Path):
    if any(s.presence=='absent' and np.any(session.labels==s.id) for s in session.segments):
        raise ValueError('已标记缺失的结构仍有掩膜，不能保存矛盾状态。')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        'schema': 2, 'app': 'Joint2STL', 'joint': session.joint, 'side': session.side,
        'roi': [[s.start, s.stop] for s in session.roi],
        'segments': [asdict(s) for s in session.segments],
        'seeds': [asdict(s) for s in session.seeds], 'parameters': session.parameters,
        'description': session.volume.description, 'series_key': session.volume.series_key,
        'reconstruction': session.volume.reconstruction,
        'modality':session.volume.modality,
        'coordinates': 'DICOM patient LPS', 'units': 'mm',
        'ai_reports': session.ai_reports,
        'preprocess_reports': session.preprocess_reports,
        'repair_reports': session.repair_reports,
    }
    # Dot-prefixed temporary files can retain UF_HIDDEN after atomic rename on
    # synchronized macOS volumes. Keep the staging name visible, with .tmp suffix.
    fd, temp = tempfile.mkstemp(prefix='joint2stl-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez_compressed(f, hu=session.volume.hu, labels=session.labels,
                                affine_lps=session.volume.affine_lps,
                                metadata=np.asarray(json.dumps(meta, ensure_ascii=False)))
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return str(path)


def load_project(path: str | Path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata'].item()))
        if meta.get('schema') not in (1,2):
            raise ValueError('不支持的项目版本。')
        hu, labels, affine = data['hu'], data['labels'], data['affine_lps']
    volume = Volume(hu, affine, meta.get('description', ''), '', meta.get('series_key', ''), meta.get('reconstruction', {}),meta.get('modality','CT'))
    if not np.isfinite(hu).all():
        raise ValueError('影像内存在无效数值。')
    segments = [Segment(**s) for s in meta['segments']]
    if any(s.kind not in ('bone','ligament','other') or s.presence not in ('unknown','present','absent') for s in segments):
        raise ValueError('项目结构类型或存在状态无效。')
    ids = [s.id for s in segments]
    if len(set(ids)) != len(ids) or any(not isinstance(i, int) or not 1 <= i <= 65535 for i in ids):
        raise ValueError('项目骨标签无效。')
    if labels.dtype != np.uint16 or labels.shape != hu.shape or not set(np.unique(labels)).issubset({0, *ids}):
        raise ValueError('项目分割与标签不一致。')
    if any(s.presence=='absent' and np.any(labels==s.id) for s in segments):
        raise ValueError('标记缺失的结构不能同时包含掩膜，请检查项目。')
    if any(len(s.color) != 7 or s.color[0] != '#' or any(c not in '0123456789abcdefABCDEF' for c in s.color[1:]) for s in segments):
        raise ValueError('标签颜色无效。')
    session = Session(volume, labels, segments, meta['joint'], meta['side'],
                      tuple(slice(*b) for b in meta['roi']))
    if session.joint not in BONES:
        raise ValueError('项目部位无效。')
    for s in meta.get('seeds', []):
        zyx = tuple(s['zyx'])
        if s['label'] not in ids or len(zyx) != 3 or any(not isinstance(i, int) or not 0 <= i < n for i, n in zip(zyx, hu.shape)):
            raise ValueError('项目种子位置无效。')
        session.seeds.append(Seed(zyx, s['label']))
    session.parameters.update(meta.get('parameters', {}))
    session.ai_reports=meta.get('ai_reports',[])
    session.preprocess_reports=meta.get('preprocess_reports',[])
    session.repair_reports=meta.get('repair_reports',[])
    session.project_path = str(path)
    return session
