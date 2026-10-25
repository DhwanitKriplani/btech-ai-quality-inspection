"""CPU material baseline. Run with Python 3.11; see README.md."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
from collections.abc import Mapping

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import models, transforms

LABELS = {4: 0, 6: 1, 9: 2}
NAMES = ["metal", "plastic", "wood"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def decode_image(value, parquet_path):
    """Prefer embedded HF bytes; fall back to a path relative to the parquet."""
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, Image.Image):
        return ImageOps.exif_transpose(value).convert("RGB")
    if isinstance(value, Mapping):
        payload = value.get("bytes")
        if payload is not None and len(payload):
            return decode_image(payload, parquet_path)
        value = value.get("path")
    if isinstance(value, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(value))) as im:
            return ImageOps.exif_transpose(im).convert("RGB")
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        if not path.is_absolute():
            path = parquet_path.parent / path
        with Image.open(path) as im:
            return ImageOps.exif_transpose(im).convert("RGB")
    raise ValueError(f"Unsupported or empty image value: {type(value).__name__}")


def resolve_files(patterns):
    import glob
    files = set()
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            raise ValueError(f"No files match: {pattern}")
        files.update(Path(p).resolve() for p in matches)
    return sorted(files)


def read_split(files, name):
    samples, fingerprints = [], set()
    for path in files:
        table = pq.read_table(path, columns=["image", "label"])
        for row_number, row in enumerate(table.to_pylist(), 1):
            label = row["label"]
            if label not in LABELS:
                continue
            try:
                image = decode_image(row["image"], path)
                gray = image.convert("L")
                digest = hashlib.sha256(str(gray.size).encode() + gray.tobytes()).hexdigest()
                if digest in fingerprints:
                    raise ValueError("Duplicate selected image within split")
                fingerprints.add(digest)
                samples.append((gray, LABELS[label]))
            except Exception as exc:
                raise ValueError(f"{name}: {path.name}, row {row_number}: {exc}") from exc
    counts = [sum(y == c for _, y in samples) for c in range(3)]
    print(f"{name}: {dict(zip(NAMES, counts))}", flush=True)
    if min(counts) == 0:
        raise ValueError(f"{name} must contain all three requested classes")
    return samples, fingerprints


def stratified_split(samples, fraction, seed):
    rng = random.Random(seed)
    train, val = [], []
    for c in range(3):
        indices = [i for i, (_, y) in enumerate(samples) if y == c]
        if len(indices) < 2:
            raise ValueError(f"Need at least two training images for {NAMES[c]}")
        rng.shuffle(indices)
        n = min(len(indices) - 1, max(1, round(len(indices) * fraction)))
        val.extend(indices[:n])
        train.extend(indices[n:])
    return train, val


def extract(samples, backbone, preprocess, batch_size):
    features = []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch = torch.stack([preprocess(im) for im, _ in samples[start:start + batch_size]])
            features.append(backbone(batch).flatten(1))
            print(f"Features: {min(start + batch_size, len(samples))}/{len(samples)}", flush=True)
    return torch.cat(features), torch.tensor([y for _, y in samples], dtype=torch.long)


def metrics(logits, labels):
    pred = logits.argmax(1)
    cm = torch.bincount(labels * 3 + pred, minlength=9).reshape(3, 3)
    tp = cm.diag().float()
    precision = tp / cm.sum(0).clamp_min(1)
    recall = tp / cm.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {"accuracy": float(tp.sum() / len(labels)), "macro_f1": float(f1.mean()),
            "confusion_matrix": cm.tolist(),
            "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                                 "f1": float(f1[i]), "support": int(cm[i].sum())}
                          for i, name in enumerate(NAMES)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", default=["data/train*.parquet"])
    parser.add_argument("--test", nargs="+", default=["data/test*.parquet"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("best_material.pt"))
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1 or min(args.epochs, args.batch_size, args.threads) < 1 or args.lr <= 0:
        parser.error("Require 0 < val-fraction < 1 and positive epochs, batch-size, threads, lr")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    train_files, test_files = resolve_files(args.train), resolve_files(args.test)
    if set(train_files) & set(test_files):
        raise ValueError("Training and test file lists overlap")
    samples, train_hashes = read_split(train_files, "Original train")
    test_samples, test_hashes = read_split(test_files, "Held-out test")
    if train_hashes & test_hashes:
        raise ValueError("Identical selected images occur in both train and test; fix split leakage")
    train_idx, val_idx = stratified_split(samples, args.val_fraction, args.seed)
    print(f"Train={len(train_idx)}, validation={len(val_idx)}, test={len(test_samples)}")
    print("CPU; frozen ImageNet ResNet18; only the 3-class head is trained.")
    print("Grayscale RGB photographs are NOT NIR measurements. Metrics below are FMD only.")
    preprocess = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    # A failed download is fatal: never silently replace pretrained weights with random ones.
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.requires_grad_(False)
    model.fc = nn.Identity()
    model.eval()
    x, y = extract(samples, model, preprocess, args.batch_size)
    head = nn.Linear(512, 3)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    best_loss, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = torch.tensor(train_idx)[torch.randperm(len(train_idx))]
        total_loss = 0.0
        for ids in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(x[ids]), y[ids])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(ids)
        head.eval()
        with torch.no_grad():
            logits = head(x[val_idx])
            val_loss = loss_fn(logits, y[val_idx]).item()
            val_acc = (logits.argmax(1) == y[val_idx]).float().mean().item()
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        print(f"Epoch {epoch:03d}/{args.epochs}: train_loss={total_loss/len(train_idx):.4f} "
              f"val_loss={val_loss:.4f} val_accuracy={val_acc:.2%}", flush=True)
    head.load_state_dict(best_state)
    model.fc = head
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": "resnet18", "model_state_dict": model.state_dict(),
                "class_names": NAMES, "original_label_to_index": LABELS,
                "pretrained_weights": "IMAGENET1K_V1", "training_mode": "frozen_backbone",
                "preprocessing": {"resize": 256, "center_crop": 224, "grayscale_channels": 3,
                                  "mean": MEAN, "std": STD},
                "seed": args.seed, "best_epoch": best_epoch, "validation_loss": best_loss,
                "train_indices": train_idx, "validation_indices": val_idx,
                "train_files": [str(p) for p in train_files],
                "test_files": [str(p) for p in test_files]}, args.output)
    # Test features and predictions are computed only after model selection and saving.
    model.fc = nn.Identity()
    tx, ty = extract(test_samples, model, preprocess, args.batch_size)
    with torch.no_grad():
        result = metrics(head(tx), ty)
    result.update(best_epoch=best_epoch, validation_loss=best_loss,
                  dataset="mcimpoi/fmd_materials grayscale RGB; not NIR")
    print("\nHeld-out FMD test metrics (rows=true, columns=predicted; metal, plastic, wood):")
    print(json.dumps(result, indent=2))
    report_path = args.output.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {args.output.resolve()}\nSaved metrics: {report_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
