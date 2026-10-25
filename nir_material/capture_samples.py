"""Save labelled camera frames on Space; no images are saved automatically."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

from PIL import Image

CLASSES = ("metal", "plastic", "wood")
HERE = Path(__file__).resolve().parent


def save_sample(gray, root, split, label, session, camera, sensor_name):
    """Save an 8-bit frame and its capture metadata, without modifying pixels."""
    if split not in ("train", "val", "test") or label not in CLASSES:
        raise ValueError("Invalid split or material label")
    image = Image.fromarray(gray)
    if image.mode != "L":
        raise ValueError("Expected an 8-bit grayscale camera frame")
    folder = Path(root) / split / label
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{session}_{time.time_ns()}.png"
    image.save(path)
    metadata = {"image": path.name, "label": label, "split": split,
                "session": session, "camera_index": camera,
                "sensor_description": sensor_name,
                "source": "camera frame; IR provenance must be checked by user",
                "captured_utc": datetime.now(timezone.utc).isoformat()}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, required=True,
                        help="Windows camera index; choose the external IR camera, not the laptop webcam")
    parser.add_argument("--data", type=Path, default=HERE / "nir_data")
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--label", choices=CLASSES, default="metal")
    parser.add_argument("--sensor-name", default="3.6mm IR 1080P CCTV module (user description)")
    args = parser.parse_args()
    if args.camera < 0:
        parser.error("camera index must be zero or greater")
    import cv2
    capture = cv2.VideoCapture(args.camera)
    window = "Collect IR samples - Space save, 1/2/3 material, Q quit"
    label, saved = args.label, 0
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    seen = set()
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open camera {args.camera}. Connect the external camera "
                               "through a supported USB/video-capture interface and check its index.")
        print(f"Camera {args.camera}; split={args.split}; output={args.data.resolve()}", flush=True)
        print("Check the preview is your IR camera. 1=metal, 2=plastic, 3=wood; Space=save; Q=quit.")
        print("Use different physical objects/capture sessions for train, val, test.")
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Cannot read a camera frame")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display = frame.copy()
            lines = [f"Camera {args.camera} | {args.split} | label={label} | saved={saved}",
                     "1 metal / 2 plastic / 3 wood | SPACE save | Q quit",
                     "Use actual IR feed. Different objects/sessions for each split."]
            for i, text in enumerate(lines):
                position = (8, 24 + i * 25)
                cv2.putText(display, text, position, cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3)
                cv2.putText(display, text, position, cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("1"), ord("2"), ord("3")):
                label = CLASSES[key - ord("1")]
            if key == 32:
                digest = hashlib.sha256(gray.tobytes()).hexdigest()
                if digest in seen:
                    print("Identical frame skipped; change the object/view before saving.", flush=True)
                    continue
                path = save_sample(gray, args.data, args.split, label, session,
                                   args.camera, args.sensor_name)
                seen.add(digest)
                saved += 1
                print(f"Saved {label}: {path}", flush=True)
    finally:
        capture.release()
        cv2.destroyAllWindows()
    print(f"Saved {saved} labelled frames. No NIR model has been trained by this capture step.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
