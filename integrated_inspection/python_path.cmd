@echo off
rem Called by the launchers; no environment activation is needed.
set "INSPECTION_PYTHON="
for %%R in ("%~dp0." "%~dp0.." "%~dp0..\..") do (
    if exist "%%~fR\.venv_gpu\Scripts\python.exe" (
        set "INSPECTION_PYTHON=%%~fR\.venv_gpu\Scripts\python.exe"
        goto :found
    )
)
for %%R in ("%~dp0." "%~dp0.." "%~dp0..\..") do (
    if exist "%%~fR\.venv\Scripts\python.exe" (
        set "INSPECTION_PYTHON=%%~fR\.venv\Scripts\python.exe"
        goto :found
    )
)
echo Python environment nahi mila. Pehle SETUP_WINDOWS.bat chalao.
exit /b 1
:found
exit /b 0
