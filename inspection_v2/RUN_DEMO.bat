@echo off
setlocal
cd /d "%~dp0"
call "%~dp0python_path.cmd"
if errorlevel 1 goto :finish
echo These are training images for a software demo, not new accuracy tests.
"%INSPECTION_PYTHON%" "%~dp0run_inspection.py" --source "%~dp0examples\scratch_rgb_train.jpg" --output "%~dp0outputs\demo_scratch" --device auto
if errorlevel 1 goto :finish
"%INSPECTION_PYTHON%" "%~dp0run_inspection.py" --source "%~dp0examples\surface_nir_train.png" --input-mode nir --output "%~dp0outputs\demo_nir" --device auto
echo Open outputs\demo_scratch and outputs\demo_nir for annotated images and reports.
:finish
echo.
pause
