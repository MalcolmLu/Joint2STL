"""RAM-derived voxel budgets, expressed as engineering estimates.

The per-voxel costs and fixed overhead are estimates, not measured peak-memory
guarantees. Computing a policy never allocates voxel arrays or probes allocation
capacity.
"""

from dataclasses import dataclass
from typing import Any


__all__ = ['SystemMemory', 'detect_memory', 'memory_policy']

_GIB = 1024 ** 3
_BASELINE_TOTAL_BYTES = 16 * _GIB
_FALLBACK_TOTAL_BYTES = 8 * _GIB
_MAX_LIMIT_VOXELS = 800_000_000
_FIXED_OVERHEAD_BYTES = 512 * 1024 ** 2
# Operation: (voxel limit at 16 GiB, estimated extra bytes per voxel).
_OPERATIONS = {
    'initial': (160_000_000, 32),
    'cleanup': (160_000_000, 32),
    'model': (100_000_000, 48),
    'split': (100_000_000, 64),
    'recovery': (100_000_000, 48),
}


@dataclass(frozen=True)
class SystemMemory:
    total_bytes: int
    available_bytes: int | None
    detected: bool = True


def _validated_bytes(value: object, *, allow_zero: bool = False) -> int:
    if (isinstance(value, bool) or not isinstance(value, int)
            or value < (0 if allow_zero else 1)):
        raise ValueError('Memory bytes must be integers with a valid sign')
    return value


def detect_memory() -> SystemMemory:
    """Read psutil RAM counters, falling back to 8 GiB if detection fails.

    Total RAM must be a positive integer; available RAM may be zero and is
    clamped to total RAM. A failed or malformed reading leaves availability
    unknown. Importing psutil lazily also permits fallback if it is unavailable.
    """
    try:
        import psutil

        snapshot = psutil.virtual_memory()
        total = _validated_bytes(snapshot.total)
        available = _validated_bytes(snapshot.available, allow_zero=True)
        return SystemMemory(total, min(available, total))
    except Exception:
        return SystemMemory(_FALLBACK_TOTAL_BYTES, None, detected=False)


def memory_policy(memory: SystemMemory) -> dict[str, Any]:
    """Return integer voxel budgets and the memory counters behind them.

    Missing availability uses physical RAM alone. Otherwise, each operation is
    bounded by the available bytes after reserve and fixed overhead. Ties retain
    the ``total`` limiting factor. Invalid explicit byte counts raise ValueError.
    """
    total = _validated_bytes(memory.total_bytes)
    available = memory.available_bytes
    if available is not None:
        available = min(_validated_bytes(available, allow_zero=True), total)

    reserved = max(_GIB, min(4 * _GIB, total // 8))
    operations = {}
    for name, (baseline_limit, bytes_per_voxel) in _OPERATIONS.items():
        physical_limit = min(
            _MAX_LIMIT_VOXELS,
            baseline_limit * total // _BASELINE_TOTAL_BYTES,
        )
        limit = physical_limit
        limiting_factor = 'total'
        if available is not None:
            available_limit = max(
                0, (available - reserved - _FIXED_OVERHEAD_BYTES) // bytes_per_voxel,
            )
            if available_limit < physical_limit:
                limit = available_limit
                limiting_factor = 'available'
        operations[name] = {
            'physical_limit_voxels': physical_limit,
            'limit_voxels': limit,
            'estimated_extra_bytes_per_voxel': bytes_per_voxel,
            'limiting_factor': limiting_factor,
        }

    return {
        'memory_total_bytes': total,
        'memory_available_bytes': available,
        'memory_detected': memory.detected,
        'operations': operations,
        'reserved_bytes': reserved,
        'fixed_overhead_bytes': _FIXED_OVERHEAD_BYTES,
    }
