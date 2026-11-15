"""Optional GPU surface filtering. Segmentation topology and mesh QA stay on CPU.

Metal implements the same separable, half-sample-reflected Gaussian as SciPy.
CUDA uses CuPy's ndimage implementation. Automatic mode safely falls back to CPU.
"""
from __future__ import annotations
from functools import lru_cache
from importlib.util import find_spec
import platform
import threading

import numpy as np
from scipy import ndimage as ndi

_lock=threading.RLock()
_metal_kernel=None
_last_backend='cpu'


@lru_cache(maxsize=1)
def available_backends():
    found={'cpu':'CPU / SciPy'}
    system=platform.system()
    if system=='Darwin' and platform.machine()=='arm64':
        try:
            import mlx.core as mx
            if mx.metal.is_available():found['metal']='Apple GPU / Metal'
        except (ImportError,RuntimeError,OSError):pass
    if system!='Darwin' and find_spec('cupy') is not None:
        if system=='Windows':
            # The two wheels can ship different minor releases of CUDA DLLs
            # with the same names. Let PyTorch load its matched libraries before
            # CuPy's import tries to preload its standalone CUDA toolkit DLLs.
            # CuPy alone remains supported when PyTorch is not installed.
            try:
                import torch
            except (ImportError,OSError,RuntimeError):
                pass
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount()>0:found['cuda']='NVIDIA GPU / CUDA'
        except Exception:pass
    return found


def last_backend():
    return _last_backend


def _metal_gaussian(field,sigma):
    import mlx.core as mx
    global _metal_kernel
    with _lock:
        mx.set_memory_limit(2*1024**3)
        mx.set_cache_limit(256*1024**2)
        if _metal_kernel is None:
            _metal_kernel=mx.fast.metal_kernel(name='joint2stl_gaussian_axis',input_names=['inp','weights','params'],output_names=['out'],source='''
                uint i = thread_position_in_grid.x;
                int count=params[0], stride=params[1], dim=params[2], radius=params[3];
                if (i >= count) return;
                int c = (i / stride) % dim;
                float value = 0.0f;
                for (int d = -radius; d <= radius; ++d) {
                    int q = c + d;
                    while (q < 0 || q >= dim) {
                        if (q < 0) q = -q - 1;
                        if (q >= dim) q = 2 * dim - q - 1;
                    }
                    int neighbor = int(i) + (q - c) * stride;
                    value += inp[neighbor] * weights[d + radius];
                }
                out[i] = value;
            ''')
        with mx.stream(mx.gpu):
            a=mx.array(np.ascontiguousarray(field,dtype=np.float32))
            for axis,s in enumerate(sigma):
                if s<=1e-12:continue
                radius=int(4*s+0.5)
                offsets=np.arange(-radius,radius+1,dtype=float)
                weights=np.exp(-0.5*(offsets/s)**2);weights/=weights.sum()
                stride=int(np.prod(field.shape[axis+1:],dtype=np.int64))
                params=mx.array([a.size,stride,field.shape[axis],radius],dtype=mx.int32)
                a=_metal_kernel(inputs=[a,mx.array(weights.astype(np.float32)),params],
                    grid=(a.size,1,1),threadgroup=(256,1,1),output_shapes=[a.shape],output_dtypes=[mx.float32])[0]
            mx.eval(a)
            result=np.array(a,copy=True)
        return result


def _cuda_gaussian(field,sigma):
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter
    from ._windows_cuda_paths import configure_nvrtc_paths
    configure_nvrtc_paths()
    with _lock:
        cp.get_default_memory_pool().set_limit(size=2*1024**3)
        a=cp.asarray(field,dtype=cp.float32)
        result=cp.asnumpy(gaussian_filter(a,sigma=tuple(sigma),mode='reflect',truncate=4))
        return result


def gaussian_filter(field,sigma,backend='auto'):
    global _last_backend
    field=np.asarray(field,dtype=np.float32)
    sigma=np.broadcast_to(np.asarray(sigma,dtype=float),(field.ndim,))
    available=available_backends()
    selected=backend
    if backend=='auto':
        # Small arrays are faster on CPU. Limit transient GPU memory on 16GB machines.
        selected=('metal' if 'metal' in available else 'cuda' if 'cuda' in available else 'cpu') if field.size>=1_000_000 and field.nbytes*4<=2*1024**3 else 'cpu'
    if selected not in available:
        raise ValueError('所选 GPU 后端不可用；可选择“自动”或 CPU，或安装对应可选依赖。')
    try:
        if selected=='metal':result=_metal_gaussian(field,sigma)
        elif selected=='cuda':result=_cuda_gaussian(field,sigma)
        else:result=ndi.gaussian_filter(field,sigma=sigma)
        _last_backend=selected
        return result
    except (RuntimeError,MemoryError,OSError):
        if backend!='auto':raise
        _last_backend='cpu'
        return ndi.gaussian_filter(field,sigma=sigma)
