"""Initialize the application-local DLL directories for the bundled Python."""
import os
from pathlib import Path
import sys


if os.name == 'nt':
    _qt_dlls = Path(sys.base_prefix, 'Lib', 'site-packages', 'PySide6')
    if _qt_dlls.is_dir():
        # Keep the directory cookie alive for DLLs loaded lazily by the wheels.
        _qt_dll_cookie = os.add_dll_directory(str(_qt_dlls))
