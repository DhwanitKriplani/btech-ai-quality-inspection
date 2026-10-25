@echo off
setlocal
cd /d "%~dp0"
:menu
cls
echo MATERIAL AND SURFACE INSPECTION
echo.
echo 1. RGB photo - material boxes and scratch detection
echo 2. RGB live camera - all RGB stages
echo 3. NIR photo - material boxes and classification
echo 4. Dual RGB + NIR cameras
echo 5. Included demo images
echo Q. Quit
echo.
choice /c 12345Q /n /m "Choose 1-5 or Q: "
if errorlevel 6 exit /b 0
if errorlevel 5 (
  call "%~dp0RUN_DEMO.bat"
  goto menu
)
if errorlevel 4 (
  call "%~dp0RUN_DUAL_CAMERA.bat"
  goto menu
)
if errorlevel 3 (
  call "%~dp0RUN_NIR_PHOTO.bat"
  goto menu
)
if errorlevel 2 (
  call "%~dp0RUN_WEBCAM.bat"
  goto menu
)
if errorlevel 1 call "%~dp0RUN_PHOTO.bat"
goto menu
