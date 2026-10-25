@echo off
setlocal
cd /d "%~dp0"
call "%~dp0python_path.cmd"
if not defined MATERIAL_PYTHON (
  py -3.11 -m venv "%~dp0.venv"
  if errorlevel 1 goto failed
  call "%~dp0python_path.cmd"
)
"%MATERIAL_PYTHON%" --version
"%MATERIAL_PYTHON%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
"%MATERIAL_PYTHON%" -m pip install numpy pillow pyarrow opencv-python requests
if errorlevel 1 goto failed
echo Setup complete. Drag a photo onto PREDICT_PHOTO.bat, or open RUN_WEBCAM.bat.
pause
exit /b 0
:failed
echo Setup failed. Read the error above. Python 3.11 must be installed.
pause
exit /b 1
