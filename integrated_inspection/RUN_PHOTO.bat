@echo off
setlocal
cd /d "%~dp0"
call "%~dp0python_path.cmd"
if errorlevel 1 goto :finish
set "INSPECTION_PHOTO=%~1"
if not defined INSPECTION_PHOTO (
    echo RGB photo ko is window mein drag-drop karo, phir Enter dabao.
    set /p "INSPECTION_PHOTO=Photo ka full path: "
)
if not defined INSPECTION_PHOTO goto :missing
set "INSPECTION_PHOTO=%INSPECTION_PHOTO:"=%"
if not exist "%INSPECTION_PHOTO%" goto :missing
echo Material boxes aur aluminium defect inspection shuru ho rahi hai...
"%INSPECTION_PYTHON%" "%~dp0run_inspection.py" --source "%INSPECTION_PHOTO%" --input-mode rgb --output "%~dp0outputs\rgb_photo" --show --device auto
goto :finish
:missing
echo Photo nahi mili. Image ka poora path do ya photo is BAT file par drag-drop karo.
:finish
echo.
pause
