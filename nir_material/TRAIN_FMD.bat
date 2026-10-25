@echo off
setlocal
cd /d "%~dp0"
call "%~dp0python_path.cmd"
if not defined MATERIAL_PYTHON (
  echo Python environment not found. Run SETUP_WINDOWS.bat first.
  pause
  exit /b 1
)
"%MATERIAL_PYTHON%" -u "%~dp0train_material.py" --train "%~dp0data\train*.parquet" --test "%~dp0data\test*.parquet" --output "%~dp0best_material.pt"
set "RUN_RESULT=%ERRORLEVEL%"
pause
exit /b %RUN_RESULT%
