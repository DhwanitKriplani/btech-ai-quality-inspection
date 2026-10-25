@echo off
setlocal
cd /d "%~dp0"
call "%~dp0python_path.cmd"
if not defined MATERIAL_PYTHON (
  echo Python environment not found. Run SETUP_WINDOWS.bat first.
  pause
  exit /b 1
)
echo Connect the external IR camera through a supported Windows video interface.
echo Camera 0 is often the laptop webcam. Check the preview before saving.
set "CAMERA=1"
set /p "CAMERA=External IR camera index [1]: "
set "SPLIT=train"
set /p "SPLIT=Split: train, val or test [train]: "
"%MATERIAL_PYTHON%" -u "%~dp0capture_samples.py" --camera "%CAMERA%" --split "%SPLIT%"
set "RUN_RESULT=%ERRORLEVEL%"
pause
exit /b %RUN_RESULT%
