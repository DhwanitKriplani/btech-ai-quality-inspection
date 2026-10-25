#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv_pi/bin/python ]]; then echo 'Run bash setup_pi.sh first.' >&2; exit 1; fi
source ./pi_config.env
mode="${1:-fixed}"
py=.venv_pi/bin/python
common=(--device cpu --threads "${THREADS:-4}" --max-materials "${MAX_MATERIALS:-4}" --stride "${STRIDE:-1}")
[[ "${SHOW:-0}" == 1 ]] && common+=(--show)
mkdir -p outputs references .runtime_cache
export YOLO_CONFIG_DIR="$PWD/.runtime_cache/ultralytics_pi"
export MPLCONFIGDIR="$PWD/.runtime_cache/mpl_pi"
case "$mode" in
 demo)
  "$py" run_inspection.py --source examples/scratch_rgb_train.jpg --output outputs/pi_demo_rgb "${common[@]}"
  "$py" run_inspection.py --source examples/surface_nir_train.png --input-mode nir --output outputs/pi_demo_nir "${common[@]}" ;;
 fixed)
  if [[ ! -f references/background_rgb.png ]]; then
   read -r -p 'Clear the inspection area completely, then press Enter to capture empty background: '
   "$py" capture_background.py --source "$RGB_SOURCE" --output references/background_rgb.png --headless
  fi
  "$py" run_inspection.py --source "$RGB_SOURCE" --background-rgb references/background_rgb.png --output outputs/pi_fixed "${common[@]}" ;;
 rgb) "$py" run_inspection.py --source "$RGB_SOURCE" --output outputs/pi_rgb "${common[@]}" ;;
 nir) "$py" run_inspection.py --source "$NIR_SOURCE" --input-mode nir --output outputs/pi_nir "${common[@]}" ;;
 dual)
  extra=();[[ -f calibration.json ]] && extra+=(--calibration calibration.json)
  "$py" run_inspection.py --source "$RGB_SOURCE" --nir-source "$NIR_SOURCE" --output outputs/pi_dual "${common[@]}" "${extra[@]}" ;;
 *) echo 'Usage: bash start_pi.sh [demo|fixed|rgb|nir|dual]' >&2; exit 2 ;;
esac
