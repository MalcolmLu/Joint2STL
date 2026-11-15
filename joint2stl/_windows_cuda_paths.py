"""Bridge the packaged NVRTC's narrow include-path handling on Windows."""
from pathlib import Path
import os
import shutil
import sys
import tempfile
import threading

_lock = threading.RLock()
_temp = None


def _private_ascii_temp():
    # TemporaryDirectory creates its leaf with mode 0700. The bundled CPython
    # 3.12.14 enforces the private Windows ACL even beneath a shared temp root.
    roots = [Path(tempfile.gettempdir())]
    for var, sub in [('LOCALAPPDATA', 'Temp'), ('SystemRoot', 'Temp'), ('PUBLIC', 'Documents')]:
        if value := os.environ.get(var):
            roots.append(Path(value) / sub)
    for root in roots:
        try:
            path = root.resolve(strict=True)
            if str(path).isascii() and path.drive and not path.drive.startswith('\\\\') and path.is_dir():
                return tempfile.TemporaryDirectory(prefix='Joint2STL-nvrtc-', dir=path, ignore_cleanup_errors=True)
        except OSError:
            continue
    raise OSError('CUDA 找不到可写入的英文临时目录。请选择 CPU 表面滤波，或将 TEMP 指向可写入的本地英文路径。')


class _CompilerTempfiles:
    """Give only CuPy's compiler an ASCII root without changing Windows TEMP."""

    def __init__(self, root):
        self.root = root

    def TemporaryDirectory(self, suffix=None, prefix=None, dir=None, **options):
        return tempfile.TemporaryDirectory(suffix=suffix, prefix=prefix, dir=self.root if dir is None else dir, **options)

    def __getattr__(self, attribute):
        return getattr(tempfile, attribute)


def configure_nvrtc_paths():
    """Adapt the packaged compiler's temporary and include directories."""
    global _temp
    if sys.platform != 'win32':
        return
    # The Windows installer uses python312._pth; source/venv installations are
    # deliberately left alone. Import only after the app initializes CUDA.
    runtime = Path(sys.base_prefix)
    if not (runtime / 'python312._pth').is_file():
        return
    unicode_headers = not str(runtime).isascii()
    unicode_temp = not tempfile.gettempdir().isascii()
    if not (unicode_headers or unicode_temp):
        return
    from cupy.cuda import compiler
    from cupy_backends.cuda.libs import nvrtc
    roots = [runtime / 'Lib/site-packages/cupy/_core/include',
             runtime / 'Lib/site-packages/nvidia/cuda_runtime/include']
    roots = tuple(r.resolve() for r in roots)
    with _lock:
        if unicode_temp and not isinstance(compiler.tempfile, _CompilerTempfiles):
            if _temp is None:
                _temp = _private_ascii_temp()
            compiler.tempfile = _CompilerTempfiles(_temp.name)
        if not unicode_headers:
            return
        original = nvrtc.compileProgram
        if getattr(original, '_joint2stl_header_bridge', False):
            return
        copied = {}

        def rewrite(option):
            global _temp
            if option.isascii():
                return option
            flag = next((p for p in ('-I', '--include-path=') if option.startswith(p)), None)
            if flag is None:
                return option
            path = Path(option[len(flag):].strip('"')).resolve()
            for index, root in enumerate(roots):
                if not path.is_relative_to(root):
                    continue
                with _lock:
                    if _temp is None:
                        _temp = _private_ascii_temp()
                    if root not in copied:
                        dst = Path(_temp.name) / f'headers{index}'
                        shutil.copytree(root, dst, dirs_exist_ok=True)
                        copied[root] = dst
                    return flag + str(copied[root] / path.relative_to(root))
            return option

        def compile_program(program, options):
            return original(program, tuple(rewrite(opt) for opt in options))

        compile_program._joint2stl_header_bridge = True
        nvrtc.compileProgram = compile_program
