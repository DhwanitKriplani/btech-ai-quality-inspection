@echo off
setlocal
cd /d "%~dp0"
call "%~dp0python_path.cmd"
if errorlevel 1 goto :finish
set "INSPECTION_PHOTO=%~1"
if not defined INSPECTION_PHOTO (
    echo NIR camera photo ko is window mein drag-drop karo, phir Enter dabao.
    set /p "INSPECTION_PHOTO=Photo ka full path: "
)
if not defined INSPECTION_PHOTO goto :missing
set "INSPECTION_PHOTO=%INSPECTION_PHOTO:"=%"
if not exist "%INSPECTION_PHOTO%" goto :missing
echo NIR material inspection shuru ho rahi hai. RGB defect model is mode mein nahi chalega.
"%INSPECTION_PYTHON%" "%~dp0run_inspection.py" --source "%INSPECTION_PHOTO%" --input-mode nir --output "%~dp0outputs\nir_photo" --show --device auto
goto :finish
:missing
echo Photo nahi mili. Image ka poora path do ya photo is BAT file par drag-drop karo.
:finish
echo.
pause
