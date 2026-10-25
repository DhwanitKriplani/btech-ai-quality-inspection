INTEGRATED APP: open integrated_inspection\START_INSPECTION.bat
Read integrated_inspection\START_HERE.txt for modes and measured limitations.
Original setup notes follow below.

VISION-BASED QUALITY INSPECTION UNIT — CV MODEL STARTER

READ THIS FIRST
This is the reproducible starter package, not a trained custom model yet. A pretrained generic model does NOT already know your four aluminium-defect classes. To train a trustworthy custom model, labelled examples of your exact classes are required. No model accuracy is guaranteed before validation on your own camera and parts.

TARGET TASKS
RGB detection/segmentation classes (fixed order):
0 Scratch
1 Black Marker
2 Corner Defect
3 Side Defect

NIR material classification classes:
Aluminium/Metal, Plastic, Wood

PHASE 1 — RGB TRAINING (run on laptop, not Raspberry Pi)
1. Install Python 3.10 or 3.11 on your laptop.
2. Open PowerShell in this extracted folder.
3. Run: py -m venv .venv
4. Run: .\.venv\Scripts\Activate.ps1
5. Run: python -m pip install --upgrade pip
6. Run: pip install -r requirements-train.txt
7. Prepare an Ultralytics-format dataset (see rgb_defects/dataset.yaml and rgb_defects/DATASET_FORMAT.txt).
8. Run: python rgb_defects/train.py --data rgb_defects/dataset.yaml --model yolo26n-seg.pt --epochs 100 --imgsz 640

The training script will stop with a clear error if the dataset path is not configured. It will not pretend that a generic pretrained model has learned your classes.

PHASE 2 — TEST RGB MODEL
python rgb_defects/predict_image.py --model runs/segment/train/weights/best.pt --source path\to\test_image.jpg
Adjust the model path to the actual run directory printed by training.

PHASE 3 — EXPORT FOR RASPBERRY PI 5
After a custom model has been trained and evaluated:
python raspi/export_ncnn.py --model runs/segment/train/weights/best.pt
Copy the generated *_ncnn_model directory to the Pi. Then follow raspi/README_RASPI.txt.

NIR WARNING
The NIR pipeline is only a scaffold until we know whether your hardware produces ordinary images or actual spectral measurements. Do not treat RGB weights as an NIR classifier. The NIR script requires a labelled dataset and is deliberately separate.
