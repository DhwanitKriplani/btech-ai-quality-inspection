"""Train on actual 8-bit NIR camera images arranged as train/val/test/class_name.

Example: python train_nir.py --data nir_data --sensor-name "my NIR camera"
Each split must contain metal/, plastic/, and wood/. Split different physical
objects/capture sessions before using this script. RGB FMD is not NIR data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import models, transforms

from train_material import MEAN, NAMES, STD, extract, metrics

HERE = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
RAW_SUFFIXES = {".raw", ".dng", ".nef", ".cr2", ".arw", ".npy", ".npz", ".hdr", ".bil", ".bsq"}
DEFAULT_PREPROCESS = {"resize": 256, "center_crop": 224, "grayscale_channels": 3,
                      "mean": MEAN, "std": STD}
MCUBES_SOURCE = "https://github.com/kyotovision-public/multimodal-material-segmentation"


def dataset_metadata(dataset_name, dataset_source, root):
    """Record the declared dataset and retain a hash of full crop provenance."""
    if dataset_name == "mcubes_nir_material_crops":
        description = "MCubeS official NIR material crops"
        dataset_source = dataset_source or MCUBES_SOURCE
        verification = "Official MCubeS NIR image channel and material annotations; see crop manifest"
    else:
        description = "user-supplied NIR camera images"
        verification = "NIR provenance supplied by user; not independently verified"
    metadata = {"dataset_name": dataset_name, "training_domain": dataset_name,
                "dataset_description": description, "dataset_source": dataset_source,
                "source_verification": verification}
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(manifest, dict):
            raise ValueError(f"Expected a JSON object in {manifest_path}")
        summary = {key: value for key, value in manifest.items() if key != "entries"}
        entries = manifest.get("entries")
        if entries is not None:
            if not isinstance(entries, list):
                raise ValueError(f"Expected an entries list in {manifest_path}")
            summary["entry_count"] = len(entries)
        metadata["dataset_manifest"] = {
            "path": str(manifest_path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
            "summary": summary,
        }
    return metadata


def read_camera_image(path):
    """Reject high-bit-depth sources rather than silently clipping intensities."""
    with Image.open(path) as im:
        if getattr(im, "n_frames", 1) != 1:
            raise ValueError("Use a single camera frame per image file")
        high_depth = im.mode in {"I", "F"} or im.mode.startswith("I;")
        if im.format == "PNG":
            with path.open("rb") as source:
                header = source.read(25)
            high_depth = high_depth or (len(header) == 25 and header[24] > 8)
        if hasattr(im, "tag_v2"):
            bits = im.tag_v2.get(258, (8,))
            if isinstance(bits, int):
                bits = (bits,)
            high_depth = high_depth or any(bit > 8 for bit in bits)
        if high_depth:
            raise ValueError("16-bit/float/raw intensities require fixed sensor normalization; "
                             "export consistently normalized 8-bit camera images first")
        if im.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "YCbCr"}:
            raise ValueError(f"Unsupported camera image mode: {im.mode}")
        return ImageOps.exif_transpose(im).convert("RGB").convert("L")


def read_folder_split(root, split, seen):
    folder = root / split
    if not folder.is_dir():
        raise ValueError(f"Missing split folder: {folder}")
    unexpected = [p.name for p in folder.iterdir() if p.is_dir() and p.name not in NAMES]
    if unexpected:
        raise ValueError(f"Unexpected class folder(s) in {folder}: {unexpected}; use {NAMES}")
    samples, files, counts = [], [], {}
    for label, name in enumerate(NAMES):
        class_dir = folder / name
        if not class_dir.is_dir():
            raise ValueError(f"Missing class folder: {class_dir}")
        paths = sorted(p for p in class_dir.rglob("*") if p.is_file())
        raw_paths = [p for p in paths if p.suffix.lower() in RAW_SUFFIXES]
        if raw_paths:
            raise ValueError(f"Raw/spectral file {raw_paths[0]}: this trainer requires "
                             "consistently normalized 8-bit camera images, not spectra")
        images = [p for p in paths if p.suffix.lower() in IMAGE_SUFFIXES]
        if not images:
            raise ValueError(f"No supported images in {class_dir}")
        for path in images:
            try:
                image = read_camera_image(path)
                fingerprint = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                if fingerprint in seen:
                    raise ValueError(f"Duplicate decoded grayscale image; first occurrence: {seen[fingerprint]}")
                seen[fingerprint] = str(path)
                samples.append((image, label))
                files.append(str(path.resolve()))
            except Exception as exc:
                raise ValueError(f"Cannot use {path}: {exc}") from exc
        counts[name] = len(images)
    print(f"{split}: {counts}", flush=True)
    return samples, files


def load_initial_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "resnet18" or checkpoint.get("class_names") != NAMES:
        raise ValueError(f"Initialization must be a resnet18 checkpoint with classes {NAMES}")
    preprocessing = checkpoint.get("preprocessing", DEFAULT_PREPROCESS)
    if preprocessing != DEFAULT_PREPROCESS:
        raise ValueError("Initialization uses unsupported preprocessing; expected resize256/crop224/grayscale3/ImageNet")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, dict(preprocessing)


def inverse_frequency_weights(labels):
    """Compute class balancing from training labels only."""
    counts = torch.bincount(labels, minlength=len(NAMES)).float()
    if len(counts) != len(NAMES) or (counts <= 0).any():
        raise ValueError("Training labels must contain all three material classes")
    return len(labels) / (len(NAMES) * counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--init", type=Path, default=HERE / "best_material.pt")
    parser.add_argument("--output", type=Path, default=HERE / "best_nir.pt")
    parser.add_argument("--sensor-name", default="unspecified")
    parser.add_argument("--dataset-name", choices=("user_supplied_nir_camera_images", "mcubes_nir_material_crops"),
                        default="user_supplied_nir_camera_images")
    parser.add_argument("--dataset-source", default="", help="Dataset source URL recorded with the model")
    parser.add_argument("--balanced-loss", action="store_true",
                        help="Use inverse training-class frequency weights for training loss only")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs, batch-size, threads and lr must be positive and finite")
    if args.output.resolve() == args.init.resolve():
        parser.error("output must differ from the initialization checkpoint")
    if args.output.name.lower() == "best_material.pt":
        parser.error("Use a distinct NIR output name; best_material.pt is the FMD baseline")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    root = args.data.resolve()
    provenance = dataset_metadata(args.dataset_name, args.dataset_source, root)
    seen, datasets, split_files = {}, {}, {}
    for split in ("train", "val", "test"):
        datasets[split], split_files[split] = read_folder_split(root, split, seen)
    print(f"Source: {provenance['dataset_description']}")
    if provenance["dataset_source"]:
        print(f"Dataset source: {provenance['dataset_source']}")
    print("CPU; frozen pretrained ResNet18; fitting only its 3-class head.", flush=True)
    model, preprocessing = load_initial_model(args.init)
    model.requires_grad_(False)
    model.fc = nn.Identity()
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(preprocessing["resize"]),
        transforms.CenterCrop(preprocessing["center_crop"]),
        transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
        transforms.Normalize(preprocessing["mean"], preprocessing["std"]),
    ])
    x, y = extract(datasets["train"], model, preprocess, args.batch_size)
    vx, vy = extract(datasets["val"], model, preprocess, args.batch_size)
    head = nn.Linear(512, len(NAMES))
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    class_weights = inverse_frequency_weights(y) if args.balanced_loss else None
    weight_values = class_weights.tolist() if class_weights is not None else None
    if weight_values is not None:
        print(f"Training class weights: {dict(zip(NAMES, weight_values))}", flush=True)
    train_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    validation_loss_fn = nn.CrossEntropyLoss()
    best_loss, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss, total_loss_weight = 0.0, 0.0
        for ids in torch.randperm(len(y)).split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = train_loss_fn(head(x[ids]), y[ids])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; check input values and learning rate")
            loss.backward()
            optimizer.step()
            batch_loss_weight = (class_weights[y[ids]].sum().item()
                                 if class_weights is not None else len(ids))
            total_loss += loss.item() * batch_loss_weight
            total_loss_weight += batch_loss_weight
        head.eval()
        with torch.no_grad():
            val_logits = head(vx)
            val_loss = validation_loss_fn(val_logits, vy).item()
            val_accuracy = (val_logits.argmax(1) == vy).float().mean().item()
        if not math.isfinite(val_loss):
            raise ValueError("Non-finite validation loss; check input values and learning rate")
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        print(f"Epoch {epoch:03d}/{args.epochs}: train_loss={total_loss/total_loss_weight:.4f} "
              f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.2%}", flush=True)
    head.load_state_dict(best_state)
    head.eval()
    model.fc = head
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": "resnet18", "model_state_dict": model.state_dict(),
                "class_names": NAMES, "preprocessing": preprocessing,
                **provenance,
                "sensor_name": args.sensor_name,
                "training_mode": "frozen_backbone", "seed": args.seed,
                "balanced_loss": args.balanced_loss, "class_weights": weight_values,
                "best_epoch": best_epoch, "validation_loss": best_loss,
                "initialization_checkpoint": str(args.init.resolve()),
                "data_files": split_files}, args.output)
    # Never extract or evaluate test features until the best model has been selected.
    model.fc = nn.Identity()
    tx, ty = extract(datasets["test"], model, preprocess, args.batch_size)
    with torch.inference_mode():
        report = metrics(head(tx), ty)
    report.update(best_epoch=best_epoch, validation_loss=best_loss,
                  dataset=args.dataset_name, sensor_name=args.sensor_name,
                  balanced_loss=args.balanced_loss, class_weights=weight_values, **provenance)
    print("Held-out test metrics (rows=true, columns=predicted; metal, plastic, wood):")
    print(json.dumps(report, indent=2))
    report_path = args.output.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {args.output.resolve()}\nSaved metrics: {report_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
