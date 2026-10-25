@echo off
setlocal
cd /d "%~dp0"
:menu
cls
echo MATERIAL AND SCRATCH INSPECTION - V2
echo.
echo 1. RGB photo - learned material regions + scratches
echo 2. Fixed-camera inspection - empty background + object boxes
echo 3. NIR photo - material classification
echo 4. Dual RGB + NIR camera views
echo 5. Included training-image demo
echo 6. Calibrate RGB / NIR views from paired photos
echo 7. Capture or update empty RGB background
echo 8. RGB live camera - learned regions, experimental
echo Q. Quit
choice /c 12345678Q /n /m "Choose 1-8 or Q: "
if errorlevel 9 exit /b 0
if errorlevel 8 (
 call "%~dp0RUN_WEBCAM.bat"
 goto menu
)
if errorlevel 7 (
 call "%~dp0CAPTURE_BACKGROUND.bat"
 goto menu
)
if errorlevel 6 (
 call "%~dp0CALIBRATE_VIEWS.bat"
 goto menu
)
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
 call "%~dp0RUN_FIXED_CAMERA.bat"
 goto menu
)
if errorlevel 1 call "%~dp0RUN_PHOTO.bat"
goto menu
