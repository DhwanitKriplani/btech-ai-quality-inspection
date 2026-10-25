RASPBERRY PI 5 DEPLOYMENT (after custom training and export)

1. Copy the exported NCNN model directory and infer_camera.py to the Pi.
2. On Raspberry Pi OS 64-bit, create an environment and install dependencies compatible with your OS/Python:
   sudo apt update
   sudo apt install -y python3-venv python3-opencv
   python3 -m venv --system-site-packages .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install ultralytics
3. Run:
   python infer_camera.py --model ./aluminium_defects_ncnn_model --camera 0

If using the Raspberry Pi CSI camera rather than a USB webcam, the capture interface may differ; we will adapt after the model has passed offline evaluation. Test CPU temperature, FPS, lighting and detection quality on actual samples before claiming live performance.
