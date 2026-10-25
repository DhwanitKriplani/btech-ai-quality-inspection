@echo off
setlocal
cd /d "%~dp0"
call "%~dp0python_path.cmd"
if errorlevel 1 goto :finish
set "RGB_CAMERA=%~1"
if not defined RGB_CAMERA set "RGB_CAMERA=0"
echo RGB camera %RGB_CAMERA% khulega. Camera window mein Q ya Esc se band karo.
"%INSPECTION_PYTHON%" "%~dp0run_inspection.py" --source "%RGB_CAMERA%" --input-mode rgb --output "%~dp0outputs\rgb_camera" --show --device auto
:finish
echo.
pause
