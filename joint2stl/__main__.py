from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

def prepare_qt_plugins():
    # Some macOS synchronized directories propagate UF_HIDDEN to wheel contents.
    # Qt's plugin enumerator skips such files, even when absolute loading works.
    if sys.platform=='darwin':
        from PySide6.QtCore import QLibraryInfo
        plugins=Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))
        cocoa=plugins/'platforms/libqcocoa.dylib'
        if cocoa.exists() and getattr(cocoa.stat(),'st_flags',0)&0x8000:
            subprocess.run(['/usr/bin/chflags','-R','nohidden',str(plugins)],check=True)

def main():
    parser=argparse.ArgumentParser(description='Joint2STL local CT workbench')
    parser.add_argument('--dicom',help='Open a DICOM folder and select a series')
    parser.add_argument('--project',help='Open an existing .j2stl project')
    args=parser.parse_args()
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication
    from .app import MainWindow
    prepare_qt_plugins()
    app=QApplication(sys.argv[:1]);app.setApplicationName('Joint2STL');app.setOrganizationName('Joint2STL')
    app.setFont(QFont('PingFang SC' if sys.platform=='darwin' else 'Microsoft YaHei',11))
    window=MainWindow();window.show()
    if args.project:QTimer.singleShot(200,lambda:window.open_project(args.project))
    elif args.dicom:QTimer.singleShot(200,lambda:window.open_dicom(args.dicom))
    return app.exec()

if __name__=='__main__':sys.exit(main())
