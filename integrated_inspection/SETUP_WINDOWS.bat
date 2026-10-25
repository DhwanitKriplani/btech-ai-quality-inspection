@echo off
setlocal
cd /d "%~dp0"
echo Portable CPU setup. Python 3.11 aur internet chahiye.
if exist "%~dp0.venv\Scripts\python.exe" goto :install
py -3.11 -m venv "%~dp0.venv"
if errorlevel 1 goto :failed
:install
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
"%~dp0.venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :failed
"%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto :failed
echo.
echo CPU setup complete. Ab RUN_PHOTO.bat ya RUN_NIR_PHOTO.bat chalao.
echo Launchers available GPU environment ko prefer karte hain.
goto :finish
:failed
echo.
echo Setup complete nahi hua. Upar wala error dekho. Python 3.11 installed hona chahiye.
:finish
pause
