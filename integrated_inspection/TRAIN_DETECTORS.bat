@echo off
setlocal
cd /d "%~dp0"
call "%~dp0python_path.cmd"
if errorlevel 1 goto :finish
echo Optional reproduction: downloads about 3.5 GB on first run and retrains detectors.
"%INSPECTION_PYTHON%" -u "%~dp0source_tools\download_mcubes.py" --include-rgb --output "%~dp0dataset_source"
if errorlevel 1 goto :finish
"%INSPECTION_PYTHON%" -u "%~dp0prepare_material_detection.py" --raw "%~dp0dataset_source\multimodal_dataset" --modality nir --output "%~dp0material_detection_data"
if errorlevel 1 goto :finish
"%INSPECTION_PYTHON%" -u "%~dp0prepare_material_detection.py" --raw "%~dp0dataset_source\multimodal_dataset" --modality rgb --output "%~dp0material_rgb_detection_data"
if errorlevel 1 goto :finish
"%INSPECTION_PYTHON%" -u "%~dp0train_material_detector.py" --data "%~dp0material_detection_data\dataset.yaml" --modality nir
if errorlevel 1 goto :finish
"%INSPECTION_PYTHON%" -u "%~dp0train_material_detector.py" --data "%~dp0material_rgb_detection_data\dataset.yaml" --modality rgb
:finish
echo.
pause
