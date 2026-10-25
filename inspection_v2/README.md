# Complete material and scratch inspection — V2

One folder contains all five trained model files and the integrated application for Raspberry Pi 5 / 4 GB and Windows. It supports material/object bounding boxes, material estimates, RGB aluminium scratch/corner/side-defect masks, photos, video, live sources and optional RGB-to-NIR calibration.

For the Pi, read **RASPBERRY_PI_START_HERE.txt**. On Windows, open **START_INSPECTION.bat**. Python environments are intentionally not bundled; inference requires no retraining.

## Raspberry Pi

Target: 64-bit Raspberry Pi OS Bookworm with Python 3.11. Inside this folder:

```bash
bash setup_pi.sh
bash start_pi.sh demo
```

Edit `pi_config.env` for USB/video-capture indices or a supported CSI source such as `picamera2:0`. Run `bash start_pi.sh fixed` for empty-background object inspection, or choose `rgb`, `nir`, or `dual`. Headless mode is the default. `SHOW=1` enables a desktop window. The profile uses CPU mode, four threads and at most three material regions per frame.

The optional Windows `UPLOAD_TO_PI.ps1` uploads the ZIP using your actual SSH login and extracts it into a new timestamped folder. It retains normal host-key checking and does not store credentials. No address was guessed and no Pi upload was performed without one.

## Bounding boxes and classification

Fixed-camera mode compares the current scene against an actual empty reference. Observed foreground changes create object boxes; material classification and RGB scratch segmentation then run together. Localization confidence is null, not a fake probability. An empty scene reports no object. Keep camera and lighting unchanged; recapture the reference after changes. Shadows or hands can also create foreground, so physical testing is required.

Without a reference, learned material-region detectors run. Their quality is still weak and they do not guarantee whole-object instances. Every box comes from model output or observed foreground geometry. There is no fixed centre square presented as detection. Detector and classifier estimates remain separate, and disagreement is shown as unsure. Unknown-material recognition is not trained.

## RGB, NIR and calibration

RGB mode integrates material regions/classification and the clean aluminium defect model. Black Marker is optional through `--include-markers` and is not counted as a physical defect. NIR mode does not apply RGB scratch weights to IR images.

Dual views remain unregistered unless a validated `calibration.json` is provided. `calibrate_views.py` uses matching photos and at least four correspondences, through an explicit desktop interface or a points JSON. It validates geometry, reprojection errors and exact frame sizes. RGB defect geometry can then be mapped into the NIR plane and associated with NIR material regions. This assumes the calibrated plane/depth and unchanged setup; it does not provide hardware synchronization or fuse confidence scores.

## Results and evidence

Photos produce annotated JPEG and JSON files. Video/live input produces MP4, per-frame JSONL and a run summary. Records contain original pixel coordinates, display transforms, hashes, model domains and association methods. The RGB classifier shares one ResNet18 backbone between true-RGB and grayscale inputs; scores remain uncalibrated.

Read MODEL_REPORT.md for actual measurements and AUDIT_CORRECTION.md for the old split contamination. Software checks cover CPU/CUDA inference, real model boxes/masks, photo/video output, batching, foreground geometry and calibration math. Included examples are clearly marked training images, not new accuracy tests.

**Actual Pi/CCTV operation and physical calibration still require hardware testing.** No Pi speed or RAM claim is made.

Official references: [Ultralytics Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/), [Raspberry Pi camera documentation](https://www.raspberrypi.com/documentation/computers/camera_software.html), [MCubeS source](https://github.com/kyotovision-public/multimodal-material-segmentation). Public dataset license and citation are in `source_tools/`.
