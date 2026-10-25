@echo off
setlocal
cd /d "%~dp0"
call "%~dp0python_path.cmd"
if errorlevel 1 goto finish
set /p "RGB=RGB photo full path: "
set /p "NIR=NIR photo full path of the SAME plane/setup: "
set "RGB=%RGB:"=%"
set "NIR=%NIR:"=%"
echo Select at least four corresponding points. Follow the window instructions.
"%INSPECTION_PYTHON%" "%~dp0calibrate_views.py" --rgb "%RGB%" --nir "%NIR%" --output "%~dp0calibration.json"
:finish
pause
