@echo off
setlocal
cd /d "%~dp0"
call "%~dp0python_path.cmd"
if errorlevel 1 goto finish
set "CAMERA=%~1"
if not defined CAMERA set "CAMERA=0"
if exist "%~dp0references\background_rgb.png" goto inspect
 echo First capture the EMPTY inspection area. Remove all objects, press SPACE.
"%INSPECTION_PYTHON%" "%~dp0capture_background.py" --camera "%CAMERA%" --output "%~dp0references\background_rgb.png"
if errorlevel 1 goto finish
:inspect
echo Now place the material in the inspection area. Q closes the result.
"%INSPECTION_PYTHON%" "%~dp0run_inspection.py" --source "%CAMERA%" --background-rgb "%~dp0references\background_rgb.png" --output "%~dp0outputs\fixed_camera" --show
:finish
pause
