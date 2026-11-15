@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install -e ".[cuda]"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --upgrade torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -e ".[ai]"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" scripts\setup_models.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" scripts\setup_mri_model.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" scripts\setup_correction_model.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -c "import torch; print('PyTorch:',torch.__version__); print('CUDA available:',torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'Check NVIDIA driver; CPU fallback is available.')"
echo NVIDIA setup finished. Restart Joint2STL and inspect Tools / Hardware.
pause
exit /b 0
:failed
echo Setup did not finish. Check Python 3.12 64-bit, network and NVIDIA driver.
pause
exit /b 1
