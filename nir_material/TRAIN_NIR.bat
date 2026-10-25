@echo off
setlocal
cd /d "%~dp0"
call "%~dp0python_path.cmd"
if not defined MATERIAL_PYTHON (
  echo Python environment not found. Run SETUP_WINDOWS.bat first.
  pause
  exit /b 1
)
echo Requires actual labelled IR camera images in nir_data/train, val and test.
"%MATERIAL_PYTHON%" -u "%~dp0train_nir.py" --data "%~dp0nir_data" --sensor-name "3.6mm IR 1080P CCTV module (user description)"
set "RUN_RESULT=%ERRORLEVEL%"
pause
exit /b %RUN_RESULT%
