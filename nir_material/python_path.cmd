@echo off
set "MATERIAL_PYTHON="
if exist "%~dp0.venv\Scripts\python.exe" set "MATERIAL_PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined MATERIAL_PYTHON if exist "%~dp0..\.venv\Scripts\python.exe" set "MATERIAL_PYTHON=%~dp0..\.venv\Scripts\python.exe"
if not defined MATERIAL_PYTHON if exist "%~dp0..\..\.venv\Scripts\python.exe" set "MATERIAL_PYTHON=%~dp0..\..\.venv\Scripts\python.exe"
if defined MATERIAL_PYTHON exit /b 0
exit /b 1
