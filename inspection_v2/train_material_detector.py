"""Train a material-region detector on scene-separated MCubeS data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--modality", choices=("rgb", "nir"), required=True)
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset yaml not found: {args.data}")
    if min(args.epochs, args.imgsz, args.batch) < 1 or args.workers < 0:
        parser.error("epochs, imgsz and batch positive; workers nonnegative")
    device = ("0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model = YOLO(args.model)
    model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, workers=args.workers, device=device, seed=42,
                deterministic=True, patience=10, pretrained=True, optimizer="AdamW",
                lr0=.001, lrf=.01, close_mosaic=5, hsv_h=0.0 if args.modality == "nir" else .015,
                hsv_s=0.0 if args.modality == "nir" else .7,
                cache=False, plots=True, project=str(HERE / "training"),
                name=f"material_{args.modality}", exist_ok=False, verbose=False)
    best = Path(model.trainer.best)
    output = HERE / "models" / f"material_{args.modality}_detector.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, output)
    selected = YOLO(str(output))
    if list(selected.names.values()) != ["metal", "plastic", "wood"]:
        raise ValueError(f"Unexpected detector class order: {selected.names}")
    # Test is not used to select epochs or adjust settings.
    evaluation = selected.val(data=str(args.data.resolve()), split="test", device=device,
                              imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                              plots=False, verbose=False, project=str(HERE / "evaluation"),
                              name=f"material_{args.modality}_test", exist_ok=False)
    report = {"model": str(output.resolve()), "modality": args.modality,
              "training_source": "MCubeS", "task": "connected material-region bounding boxes",
              "names": selected.names, "trained_epochs_requested": args.epochs,
              "run_directory": str(best.parent.parent.resolve()),
              "held_out_test": {"precision": float(evaluation.box.mp),
                                "recall": float(evaluation.box.mr),
                                "map50": float(evaluation.box.map50),
                                "map50_95": float(evaluation.box.map),
                                "per_class_map50_95": dict(zip(selected.names.values(), map(float, evaluation.box.maps)))},
              "weights_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "limits": "Region boxes are not guaranteed physical object instances; user CCTV setup untested."}
    report_path = HERE / "models" / f"material_{args.modality}_detector.metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("SAVED MATERIAL DETECTOR:", output.resolve(), flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
