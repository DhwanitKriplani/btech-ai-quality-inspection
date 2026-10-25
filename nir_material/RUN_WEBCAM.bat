@echo off
setlocal
cd /d "%~dp0"
call "%~dp0python_path.cmd"
if not defined MATERIAL_PYTHON (
  echo Python environment not found. Run SETUP_WINDOWS.bat first.
  pause
  exit /b 1
)
set "CAMERA=0"
if not "%~1"=="" set "CAMERA=%~1"
echo Keep one material in the centre. Q or Escape closes the camera window.
set "MATERIAL_MODEL=%~dp0best_material.pt"
if exist "%~dp0best_nir.pt" set "MATERIAL_MODEL=%~dp0best_nir.pt"
echo Selected model: %MATERIAL_MODEL%
"%MATERIAL_PYTHON%" -u "%~dp0predict_material.py" --model "%MATERIAL_MODEL%" --camera "%CAMERA%"
set "RUN_RESULT=%ERRORLEVEL%"
pause
exit /b %RUN_RESULT%
