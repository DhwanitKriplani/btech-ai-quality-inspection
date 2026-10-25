#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
case "$(uname -m)" in aarch64|arm64) ;; *) echo "Use 64-bit Raspberry Pi OS (aarch64)." >&2; exit 1;; esac
python3 -c "import sys; assert sys.version_info[:2]==(3,11), 'This profile targets Raspberry Pi OS 64-bit Bookworm with Python3.11; do not install mismatched system camera packages.'"
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip python3-opencv python3-picamera2 ffmpeg libgl1 libglib2.0-0
python3 -m venv --system-site-packages .venv_pi
.venv_pi/bin/python -m pip install --upgrade pip
.venv_pi/bin/python -m pip install -r requirements_pi.txt
.venv_pi/bin/python pi_doctor.py
printf '\nSetup finished. Run: bash start_pi.sh demo\n'
