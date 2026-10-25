@echo off
setlocal
cd /d "%~dp0"
call "%~dp0python_path.cmd"
if not defined MATERIAL_PYTHON (
  echo Python environment not found. Run SETUP_WINDOWS.bat first.
  pause
  exit /b 1
)
set "PHOTO=%~1"
if not defined PHOTO set /p "PHOTO=Photo path paste karo, phir Enter: "
set "PHOTO=%PHOTO:"=%"
if not defined PHOTO (
  echo No photo path supplied.
  pause
  exit /b 1
)
set "MATERIAL_MODEL=%~dp0best_material.pt"
if exist "%~dp0best_nir.pt" set "MATERIAL_MODEL=%~dp0best_nir.pt"
"%MATERIAL_PYTHON%" -u "%~dp0predict_material.py" --model "%MATERIAL_MODEL%" --source "%PHOTO%" --output "%~dp0outputs\prediction.jpg" --json "%~dp0outputs\prediction.json"
set "RUN_RESULT=%ERRORLEVEL%"
if "%RUN_RESULT%"=="0" echo Result saved in the outputs folder.
pause
exit /b %RUN_RESULT%
