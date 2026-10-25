@echo off
setlocal
cd /d "%~dp0"
call "%~dp0python_path.cmd"
if errorlevel 1 goto :finish
set "RGB_CAMERA=%~1"
set "NIR_CAMERA=%~2"
if not defined RGB_CAMERA set "RGB_CAMERA=0"
if not defined NIR_CAMERA set "NIR_CAMERA=1"
echo RGB aur NIR cameras khulenge. Q ya Esc se band karo.
echo Dono camera views alag panes hain; unki positions registered ya aligned nahi hain.
"%INSPECTION_PYTHON%" "%~dp0run_inspection.py" --source "%RGB_CAMERA%" --input-mode rgb --nir-source "%NIR_CAMERA%" --output "%~dp0outputs\dual_camera" --show --device auto
:finish
echo.
pause
