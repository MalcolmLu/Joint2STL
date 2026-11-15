"""Deterministic memory-counter tests; no clinical data or voxel allocations."""

import builtins
from types import SimpleNamespace

import psutil
import pytest

from joint2stl.resources import SystemMemory, detect_memory, memory_policy


GIB = 1024 ** 3
BYTES_PER_VOXEL = {
    'initial': 32, 'cleanup': 32, 'model': 48, 'split': 64, 'recovery': 48,
}


@pytest.mark.parametrize('total_gib, initial_limit, other_limit, reserve_gib', [
    (16, 160_000_000, 100_000_000, 2),
    (32, 320_000_000, 200_000_000, 4),
    (64, 640_000_000, 400_000_000, 4),
    (80, 800_000_000, 500_000_000, 4),
    (256, 800_000_000, 800_000_000, 4),
])
def test_high_availability_scales_physical_limits_and_caps_each_operation(
    total_gib, initial_limit, other_limit, reserve_gib,
):
    policy = memory_policy(SystemMemory(total_gib * GIB, total_gib * GIB))

    assert policy == {
        'memory_total_bytes': total_gib * GIB,
        'memory_available_bytes': total_gib * GIB,
        'memory_detected': True,
        'operations': {
            name: {
                'physical_limit_voxels': initial_limit if name in ('initial', 'cleanup') else other_limit,
                'limit_voxels': initial_limit if name in ('initial', 'cleanup') else other_limit,
                'estimated_extra_bytes_per_voxel': cost,
                'limiting_factor': 'total',
            }
            for name, cost in BYTES_PER_VOXEL.items()
        },
        'reserved_bytes': reserve_gib * GIB,
        'fixed_overhead_bytes': 512 * 1024 ** 2,
    }


def test_low_availability_on_32_gib_machine_lowers_each_stage_budget():
    policy = memory_policy(SystemMemory(32 * GIB, 8 * GIB))
    expected_limits = {
        'initial': 117_440_512,
        'cleanup': 117_440_512,
        'model': 78_293_674,
        'split': 58_720_256,
        'recovery': 78_293_674,
    }

    assert policy['reserved_bytes'] == 4 * GIB
    for name, limit in expected_limits.items():
        operation = policy['operations'][name]
        assert operation['physical_limit_voxels'] == (
            320_000_000 if name in ('initial', 'cleanup') else 200_000_000
        )
        assert operation['limit_voxels'] == limit
        assert operation['limiting_factor'] == 'available'


def test_small_machine_keeps_one_gib_reserve():
    policy = memory_policy(SystemMemory(4 * GIB, 2 * GIB))

    assert policy['reserved_bytes'] == GIB
    assert {name: operation['limit_voxels'] for name, operation in policy['operations'].items()} == {
        'initial': 16_777_216,
        'cleanup': 16_777_216,
        'model': 11_184_810,
        'split': 8_388_608,
        'recovery': 11_184_810,
    }


@pytest.mark.parametrize('available', [0, GIB, 9 * GIB // 2 - 1, 9 * GIB // 2, 9 * GIB // 2 + 31])
def test_insufficient_headroom_has_zero_budget(available):
    policy = memory_policy(SystemMemory(32 * GIB, available))

    for operation in policy['operations'].values():
        assert operation['limit_voxels'] == 0
        assert operation['limiting_factor'] == 'available'


def test_headroom_is_rounded_down_to_whole_voxels():
    policy = memory_policy(SystemMemory(32 * GIB, 9 * GIB // 2 + 32))

    assert {name: operation['limit_voxels'] for name, operation in policy['operations'].items()} == {
        'initial': 1, 'cleanup': 1, 'model': 0, 'split': 0, 'recovery': 0,
    }


@pytest.mark.parametrize('available, expected_limit, expected_factor', [
    (7_804_354_559, 159_999_999, 'available'),
    (7_804_354_560, 160_000_000, 'total'),
    (7_804_354_561, 160_000_000, 'total'),
])
def test_availability_limits_only_when_lower_than_physical_budget(
    available, expected_limit, expected_factor,
):
    policy = memory_policy(SystemMemory(16 * GIB, available))

    assert policy['operations']['initial']['limit_voxels'] == expected_limit
    assert policy['operations']['initial']['limiting_factor'] == expected_factor
    # At the same availability, the higher-cost split stage is still limited.
    assert policy['operations']['split']['limiting_factor'] == 'available'


def test_missing_availability_uses_only_physical_limits():
    policy = memory_policy(SystemMemory(32 * GIB, None))

    assert policy['memory_available_bytes'] is None
    assert policy['memory_detected'] is True
    assert policy['reserved_bytes'] == 4 * GIB
    for name, operation in policy['operations'].items():
        expected = 320_000_000 if name in ('initial', 'cleanup') else 200_000_000
        assert operation['physical_limit_voxels'] == expected
        assert operation['limit_voxels'] == expected
        assert operation['limiting_factor'] == 'total'


def test_fractional_baseline_uses_integer_floor():
    policy = memory_policy(SystemMemory(16 * GIB - 1, None))

    assert policy['reserved_bytes'] == 2 * GIB - 1
    assert policy['operations']['initial']['limit_voxels'] == 159_999_999
    assert policy['operations']['model']['limit_voxels'] == 99_999_999
    assert all(type(operation['limit_voxels']) is int for operation in policy['operations'].values())


def test_policy_clamps_available_to_total():
    policy = memory_policy(SystemMemory(32 * GIB, 64 * GIB))

    assert policy['memory_available_bytes'] == 32 * GIB
    assert policy == memory_policy(SystemMemory(32 * GIB, 32 * GIB))


@pytest.mark.parametrize('available, expected', [
    (12 * GIB, 12 * GIB), (0, 0), (64 * GIB, 32 * GIB),
])
def test_detect_memory_reads_counters_and_clamps_available(monkeypatch, available, expected):
    monkeypatch.setattr(
        psutil, 'virtual_memory', lambda: SimpleNamespace(total=32 * GIB, available=available),
    )

    assert detect_memory() == SystemMemory(32 * GIB, expected, detected=True)


def assert_fallback(memory):
    assert memory == SystemMemory(8 * GIB, None, detected=False)
    policy = memory_policy(memory)
    assert policy['memory_total_bytes'] == 8 * GIB
    assert policy['memory_available_bytes'] is None
    assert policy['memory_detected'] is False
    assert policy['reserved_bytes'] == GIB
    for name, operation in policy['operations'].items():
        expected = 80_000_000 if name in ('initial', 'cleanup') else 50_000_000
        assert operation['physical_limit_voxels'] == expected
        assert operation['limit_voxels'] == expected
        assert operation['limiting_factor'] == 'total'


@pytest.mark.parametrize('error', [OSError('unavailable'), RuntimeError('unavailable')])
def test_detection_failure_falls_back(monkeypatch, error):
    def fail():
        raise error

    monkeypatch.setattr(psutil, 'virtual_memory', fail)

    assert_fallback(detect_memory())


def test_missing_psutil_falls_back(monkeypatch):
    original_import = builtins.__import__

    def without_psutil(name, *args, **kwargs):
        if name == 'psutil':
            raise ImportError('psutil is unavailable')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_psutil)

    assert_fallback(detect_memory())


@pytest.mark.parametrize('field, value', [
    ('total', 0), ('total', -1), ('total', None), ('total', True),
    ('total', 16.5), ('total', '17179869184'),
    ('total', float('inf')), ('total', float('nan')),
    ('available', -1), ('available', None), ('available', False),
    ('available', 1.5), ('available', '1073741824'),
    ('available', float('inf')), ('available', float('nan')),
])
def test_invalid_detected_values_fall_back(monkeypatch, field, value):
    counters = {'total': 32 * GIB, 'available': 16 * GIB}
    counters[field] = value
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(**counters))

    assert_fallback(detect_memory())


@pytest.mark.parametrize('counters', [{}, {'total': 32 * GIB}, {'available': 16 * GIB}])
def test_missing_detected_counters_fall_back(monkeypatch, counters):
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(**counters))

    assert_fallback(detect_memory())
