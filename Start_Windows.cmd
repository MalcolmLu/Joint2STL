@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo Install Python 3.12 64-bit from python.org, then run this file again.
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -c "import PySide6.QtWidgets,pydicom,scipy,skimage,trimesh,vtk,psutil,joint2stl" >nul 2>&1
if errorlevel 1 (
  ".venv\Scripts\python.exe" -m pip install -e .
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -m joint2stl %*
if errorlevel 1 pause
