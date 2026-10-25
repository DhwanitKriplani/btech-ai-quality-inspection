@echo off
setlocal
cd /d "%~dp0"
call "%~dp0python_path.cmd"
if errorlevel 1 goto finish
set "CAMERA=%~1"
if not defined CAMERA set "CAMERA=0"
set "MODE=%~2"
if not defined MODE set "MODE=rgb"
echo Empty inspection area rakho. SPACE saves background, Q cancels.
"%INSPECTION_PYTHON%" "%~dp0capture_background.py" --camera "%CAMERA%" --mode "%MODE%" --output "%~dp0references\background_%MODE%.png" --overwrite
:finish
pause
