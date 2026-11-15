"""Anatomical naming from physical LPS geometry, with explicit abstention.

These rules are not a trained semantic segmentation network. They operate on
existing components, never change bone masks and never mark a name reviewed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .model import Volume


@dataclass
class BoneGeometry:
    ident: int
    volume: float
    center: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    std: np.ndarray

    @property
    def span(self):
        return self.upper - self.lower


def component_geometry(volume: Volume, labels, offset=(0, 0, 0)):
    output = []
    for ident, box in enumerate(ndi.find_objects(labels), 1):
        if box is None:
            continue
        mask = labels[box] == ident
        flat = np.flatnonzero(mask)
        if not len(flat):
            continue
        sample = flat[::max(1, int(np.ceil(len(flat) / 24000)))]
        points = np.column_stack(np.unravel_index(sample, mask.shape))
        points += np.array([s.start for s in box]) + np.asarray(offset)
        world = volume.world(points)
        output.append(BoneGeometry(ident, len(flat) * volume.voxel_mm3, world.mean(0),
                                   world.min(0), world.max(0), world.std(0)))
    return output


def _side_groups(features, requested_side):
    ordered = sorted(features, key=lambda f: f.center[0])
    total = sum(f.volume for f in ordered)
    best = None
    for cut in range(2, len(ordered) - 1):
        right, left = ordered[:cut], ordered[cut:]
        gap = left[0].center[0] - right[-1].center[0]
        weights = [sum(f.volume for f in group) for group in (right, left)]
        spreads = [np.ptp([f.center[0] for f in group]) for group in (right, left)]
        balanced = min(weights) > total * .18
        cropped_opposite_leg = min(weights) > total * .04 and gap > max(90, 1.5 * max(spreads))
        if gap > max(45, .8 * max(spreads)) and (balanced or cropped_opposite_leg):
            if best is None or gap > best[0]:
                best = (gap, right, left)
    if best:
        return [('右侧', best[1]), ('左侧', best[2])]
    side = requested_side if requested_side in ('左侧', '右侧') else '侧别待定'
    return [(side, features)]


def _proposal(feature, side, bone, evidence):
    return {'id': feature.ident, 'name': side + '_' + bone, 'bone': bone, 'side': side,
            'method': 'lps_geometry_v1', 'status': 'identified' if side != '侧别待定' else 'check_side',
            'evidence': evidence, 'expert_confirmed': False}


def _knee(group, side):
    if len(group) < 2:
        return _connected_knee_hint(group, side)
    ordered = sorted(group, key=lambda f: f.volume, reverse=True)
    first, second = ordered[:2]
    femur, tibia = sorted((first, second), key=lambda f: f.center[2], reverse=True)
    # Two large, vertically adjacent long bones establish the joint. A single
    # tibia plus a small fibula must not be forced into a femur/tibia pair.
    if (second.volume < first.volume * .18 or min(femur.span[2], tibia.span[2]) < 18
            or femur.center[2] - tibia.center[2] < 12
            or femur.lower[2] < tibia.lower[2] + 5
            or femur.upper[2] < tibia.upper[2] + 5
            or abs(femur.lower[2] - tibia.upper[2]) > 35
            or abs(femur.center[0] - tibia.center[0]) > max(femur.span[0], tibia.span[0])):
        return _connected_knee_hint(group, side)
    result = [_proposal(femur, side, '股骨', ['主长骨位于关节近端', '与远端主长骨纵向相邻']),
              _proposal(tibia, side, '胫骨', ['主长骨位于关节远端', '与近端主长骨纵向相邻'])]
    others = [f for f in group if f.ident not in (femur.ident, tibia.ident)]
    patella_candidates = []
    fibula_candidates = []
    lateral_sign = -1 if side == '右侧' else 1 if side == '左侧' else 0
    for feature in others:
        if (feature.volume < femur.volume * .55 and feature.span[2] < 85
                and feature.center[1] < min(femur.center[1], tibia.center[1]) - 8
                and femur.lower[2] - 12 < feature.center[2] < femur.upper[2] + 12
                and feature.span[1] < max(35, femur.span[1] * .85)):
            score = (femur.center[1] - feature.center[1]) * np.cbrt(feature.volume)
            patella_candidates.append((score, feature))
        lateral = (feature.center[0] - tibia.center[0]) * lateral_sign
        if (feature.volume < tibia.volume * .5 and feature.span[2] > 20
                and feature.span[2] > 1.2 * max(feature.span[:2])
                and feature.upper[2] < tibia.upper[2] + 8
                and feature.center[1] > tibia.center[1] - 8
                and (lateral > 8 if lateral_sign else abs(feature.center[0] - tibia.center[0]) > 8)):
            fibula_candidates.append((feature.span[2] * np.cbrt(feature.volume), feature))
    if patella_candidates:
        patella = max(patella_candidates, key=lambda entry: entry[0])[1]
        result.append(_proposal(patella, side, '髌骨', ['位于关节前方', '前后方向较扁、纵向范围有限']))
    if fibula_candidates:
        fibula = max(fibula_candidates, key=lambda entry: entry[0])[1]
        if all(r['id'] != fibula.ident for r in result):
            result.append(_proposal(fibula, side, '腓骨', ['胫骨外侧的细长骨', '近端低于胫骨平台']))
    return result


def _connected_knee_hint(group, side):
    if not group:
        return []
    main = max(group, key=lambda f:f.volume)
    if main.span[2] < 100 or main.span[2] < 1.4 * max(main.span[:2]):
        return []
    anterior = [f for f in group if f.ident != main.ident and 700 < f.volume < main.volume * .3
                and f.span[2] < 85 and f.span[1] < 40 and f.center[1] < main.center[1] - 10
                and main.lower[2] + 25 < f.center[2] < main.upper[2] - 25]
    if not anterior:
        return []
    patella = max(anterior, key=lambda f:f.volume)
    connected = _proposal(main, side, '股骨/胫骨区域 · 疑似相连', ['主长骨跨越髌骨所在的关节区域', '当前几何不足以分开命名，请先分离并复核'])
    connected['status'] = 'check_connection'
    return [connected, _proposal(patella, side, '髌骨', ['位于关节前方', '形态及位置符合髌骨候选'])]


def _shoulder(group, side):
    if len(group) < 2 or side == '侧别待定':
        return []
    lateral_sign = -1 if side == '右侧' else 1
    # Humerus: lateral long axis close to superior/inferior. The scapula is a
    # posterior plate and the clavicle is a superior/anterior transverse strut.
    humeri = [f for f in group if f.span[2] > 35 and f.std[2] > 1.25 * max(f.std[:2])]
    if not humeri:
        return []
    humerus = max(humeri, key=lambda f: (f.center[0] * lateral_sign, f.volume))
    medial = [f for f in group if f.ident != humerus.ident
              and (humerus.center[0] - f.center[0]) * lateral_sign > 8]
    scapulae = [f for f in medial if f.volume > humerus.volume * .12
                and f.span[2] > 35 and f.span[0] > 25 and f.center[1] > humerus.center[1] + 5]
    if not scapulae:
        return []
    scapula = max(scapulae, key=lambda f: f.volume)
    clavicles = [f for f in medial if f.ident != scapula.ident
                 and f.volume < max(humerus.volume, scapula.volume) * .65
                 and f.span[0] > 30 and f.std[0] > 1.25 * f.std[2]
                 and f.center[1] < scapula.center[1] - 8
                 and f.center[2] > humerus.center[2] + 5]
    result = [_proposal(humerus, side, '肱骨', ['外侧纵向主长骨']),
              _proposal(scapula, side, '肩胛骨', ['位于肱骨内后方', '覆盖较大纵向与横向范围'])]
    if clavicles:
        clavicle = max(clavicles, key=lambda f: f.span[0])
        result.append(_proposal(clavicle, side, '锁骨', ['内上方的横向细长骨', '位于肩胛骨前方']))
    return result


def _flag_connected_lower_leg(volume, labels, proposal, offset):
    if proposal['bone'] != '胫骨':
        return
    mask = labels == proposal['id']
    occupied = np.flatnonzero(mask.any(axis=(1, 2)))
    if not len(occupied):
        return
    superior = volume.world(np.column_stack((occupied + offset[0],
                                             np.full(len(occupied), labels.shape[1] / 2 + offset[1]),
                                             np.full(len(occupied), labels.shape[2] / 2 + offset[2]))))[:, 2]
    checked = occupied[superior < superior.max() - 12]
    separate = 0
    sign = -1 if proposal['side'] == '右侧' else 1 if proposal['side'] == '左侧' else 0
    for z in checked:
        components, _ = ndi.label(mask[z], structure=np.ones((3, 3), bool))
        counts = np.bincount(components.ravel()); counts[0] = 0
        largest = np.argsort(counts)[-2:]
        if len(largest) < 2:
            continue
        small, large = largest
        if counts[small] * np.prod(volume.spacing_zyx[1:]) < 6 or not .025 < counts[small] / max(counts[large], 1) < .45:
            continue
        centers = ndi.center_of_mass(mask[z], components, [small, large])
        world = volume.world([[z + offset[0], y + offset[1], x + offset[2]] for y, x in centers])
        delta = world[0] - world[1]
        if (delta[0] * sign > 12 if sign else abs(delta[0]) > 12) and delta[1] > -10:
            separate += 1
    if separate * volume.spacing_zyx[0] >= max(15, len(occupied) * volume.spacing_zyx[0] * .2):
        proposal['status'] = 'check_connection'
        proposal['name'] += '区域 · 疑似相连'
        proposal['evidence'].append('远端多层存在外侧独立细骨，疑似胫腓骨共用标签，请分离后重新识别')


def identify_bones(volume, labels, joint='knee', side='双侧', offset=(0, 0, 0), progress=lambda *_: None):
    if joint not in ('knee', 'shoulder'):
        raise ValueError('不支持的建模部位。')
    progress(10, '分析骨块的物理位置与形态')
    features = component_geometry(volume, labels, offset)
    # Very small fragments are deliberately left unresolved, rather than using
    # them to determine the side or forcing a complete set of anatomical names.
    substantial = [f for f in features if f.volume >= 300
                   and not (f.span[1] < 12 and f.span[0] > 120 and f.span[0] > 8 * f.span[1])]
    proposals = []
    for detected_side, group in _side_groups(substantial, side):
        proposals.extend((_knee if joint == 'knee' else _shoulder)(group, detected_side))
    for proposal in proposals:
        _flag_connected_lower_leg(volume, labels, proposal, offset)
    named = {r['id'] for r in proposals}
    for feature in features:
        if feature.ident not in named:
            proposals.append({'id': feature.ident, 'name': '', 'bone': '', 'side': '',
                              'method': 'lps_geometry_v1', 'status': 'uncertain',
                              'evidence': ['骨块不完整、相连或位置形态不足以可靠区分'], 'expert_confirmed': False})
    progress(100, '骨标签识别完成')
    return proposals
