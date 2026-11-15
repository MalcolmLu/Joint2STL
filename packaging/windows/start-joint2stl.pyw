"""Launch the installed Joint2STL without a console window."""
import ctypes
import os
from pathlib import Path
import traceback


def main():
    user_root = Path(os.environ['LOCALAPPDATA']) / 'Joint2STL'
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('CUPY_CACHE_DIR', str(user_root / 'Cache' / 'CuPy'))
    try:
        from joint2stl.__main__ import main as workbench
        return workbench()
    except Exception as error:
        msg = f'Joint2STL 无法启动。\n\n{type(error).__name__}: {error}'
        try:
            user_root.mkdir(parents=True, exist_ok=True)
            log = user_root / 'startup-error.log'
            log.write_text(traceback.format_exc(), encoding='utf-8')
            msg += f'\n\n启动日志：{log}'
        except OSError:
            pass
        ctypes.windll.user32.MessageBoxW(None, msg, 'Joint2STL 0.4.4', 0x10)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
