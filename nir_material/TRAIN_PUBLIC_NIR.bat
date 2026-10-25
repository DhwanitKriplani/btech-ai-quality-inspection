@echo off
setlocal
cd /d "%~dp0"
call "%~dp0python_path.cmd"
if not defined MATERIAL_PYTHON (
  echo Python environment not found. Run SETUP_WINDOWS.bat first.
  pause
  exit /b 1
)
echo Reproducing public MCubeS NIR model. First run downloads about 800 MB.
"%MATERIAL_PYTHON%" -u "%~dp0download_mcubes.py"
if errorlevel 1 goto failed
"%MATERIAL_PYTHON%" -u "%~dp0prepare_mcubes_crops.py" --raw "%~dp0mcubes_raw\multimodal_dataset" --output "%~dp0mcubes_crops"
if errorlevel 1 goto failed
"%MATERIAL_PYTHON%" -u "%~dp0train_nir.py" --data "%~dp0mcubes_crops" --dataset-name mcubes_nir_material_crops --balanced-loss --sensor-name "FLIR GS3-U3-41C6NIR-C; 750 nm long-pass"
if errorlevel 1 goto failed
echo Public NIR training complete. See best_nir.pt and best_nir.metrics.json.
pause
exit /b 0
:failed
echo Training failed. Read the error above.
pause
exit /b 1
