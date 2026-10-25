# BTech AI quality inspection

The latest integrated release is in **[inspection_v2](inspection_v2/README.md)**. It includes five trained models, material/object bounding boxes, material classification, RGB aluminium scratch/corner/side-defect segmentation, Windows launchers and Raspberry Pi 5 setup scripts.

## Download the models

Model weights use Git LFS. GitHub's source ZIP may contain pointers instead of the actual weights, so clone with Git LFS installed:

```bash
git lfs install
git clone https://github.com/DhwanitKriplani/btech-ai-quality-inspection.git
cd btech-ai-quality-inspection
git lfs pull
```

For an existing clone, run `git pull --ff-only` followed by `git lfs pull`.

## Windows

Open `inspection_v2/SETUP_WINDOWS.bat` if no compatible Python environment is installed, then open `inspection_v2/START_INSPECTION.bat`. Read [START_HERE.txt](inspection_v2/START_HERE.txt) for the available modes.

## Raspberry Pi 5 / 4 GB

Use 64-bit Raspberry Pi OS Bookworm with Python 3.11. Install Git LFS before cloning:

```bash
sudo apt update
sudo apt install -y git git-lfs
git lfs install
git clone https://github.com/DhwanitKriplani/btech-ai-quality-inspection.git
cd btech-ai-quality-inspection
git lfs pull
cd inspection_v2
bash setup_pi.sh
bash start_pi.sh demo
```

Read [RASPBERRY_PI_START_HERE.txt](inspection_v2/RASPBERRY_PI_START_HERE.txt) to configure USB/CSI sources and run fixed-camera inspection. The Windows uploader requires a separately supplied release ZIP; cloning the repo on the Pi is sufficient without that ZIP.

## Measured results and remaining hardware work

Actual Pi/CCTV operation and physical calibration still need hardware testing. Learned material-region detectors have weak measured accuracy; fixed-camera mode instead localizes changes against an empty reference. Material predictions, defect masks and camera associations retain their documented limits.

See [MODEL_REPORT.md](inspection_v2/MODEL_REPORT.md), [AUDIT_CORRECTION.md](inspection_v2/AUDIT_CORRECTION.md) and the [evaluation evidence](inspection_v2/evaluation). RGB-to-grayscale data is not presented as NIR validation. No Pi speed, RAM or unseen-camera accuracy is claimed.

Earlier project folders are retained alongside V2.
