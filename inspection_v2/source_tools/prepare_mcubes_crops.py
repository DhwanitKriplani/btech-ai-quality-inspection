"""Prepare genuine MCubeS NIR material crops, keeping test scenes out of training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

LABELS = {"metal": 2, "plastic": 7, "wood": 15}
SOURCE = "https://github.com/kyotovision-public/multimodal-material-segmentation"
SIZES = (256, 192, 128, 96, 64, 48)


def scene(image_id):
    return image_id.rsplit("_", 1)[0]


def fixed_uint8(image):
    if image.ndim != 2:
        raise ValueError("Expected one-channel NIR camera image")
    if image.dtype == np.uint16:
        return np.rint(image.astype(np.float32) / 257.0).astype(np.uint8)
    if image.dtype == np.uint8:
        return image
    raise ValueError(f"Unsupported NIR dtype {image.dtype}")


def sums(integral, ys, xs, side):
    return (integral[(ys + side)[:, None], (xs + side)[None, :]]
            - integral[ys[:, None], (xs + side)[None, :]]
            - integral[(ys + side)[:, None], xs[None, :]]
            + integral[ys[:, None], xs[None, :]])


def overlap(a, b):
    ax, ay, side = a
    bx, by, other = b
    intersection = max(0, min(ax + side, bx + other) - max(ax, bx)) * max(0, min(ay + side, by + other) - max(ay, by))
    return intersection / (side * side + other * other - intersection)


def candidate_crops(gt, valid, label, maximum, purity, rng):
    ci = cv2.integral((gt == label).astype(np.uint8), sdepth=cv2.CV_64F)
    vi = cv2.integral(valid.astype(np.uint8), sdepth=cv2.CV_64F)
    height, width = gt.shape
    chosen = []
    for side in SIZES:
        if min(height, width) < side:
            continue
        step = max(8, side // 4)
        ys = np.arange(0, height - side + 1, step)
        xs = np.arange(0, width - side + 1, step)
        whole_purity = sums(ci, ys, xs, side) / side**2
        whole_valid = sums(vi, ys, xs, side) / side**2
        centre_side = round(side * 224 / 256)
        offset = (side - centre_side) // 2
        centre_purity = sums(ci, ys + offset, xs + offset, centre_side) / centre_side**2
        centre_valid = sums(vi, ys + offset, xs + offset, centre_side) / centre_side**2
        passing = np.argwhere((whole_purity >= purity) & (centre_purity >= purity)
                              & (whole_valid >= .99) & (centre_valid >= .99))
        # Seeded spatial order spreads equal-size samples, independent of model outcomes.
        rng.shuffle(passing)
        for row, col in passing:
            box = (int(xs[col]), int(ys[row]), side)
            if any(overlap(box, item[0]) > .2 for item in chosen):
                continue
            chosen.append((box, float(whole_purity[row, col]), float(centre_purity[row, col]),
                           float(whole_valid[row, col]), float(centre_valid[row, col])))
            if len(chosen) >= maximum:
                return chosen
    return chosen


def read_splits(raw):
    originals = {s: [x.strip() for x in (raw / "list_folder" / f"{s}.txt").read_text().splitlines() if x.strip()]
                 for s in ("train", "val", "test")}
    all_ids = [i for ids in originals.values() for i in ids]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Source image IDs repeat within/across official split lists")
    test_scenes = {scene(i) for i in originals["test"]}
    kept = {"test": originals["test"]}
    kept["val"] = [i for i in originals["val"] if scene(i) not in test_scenes]
    val_scenes = {scene(i) for i in kept["val"]}
    kept["train"] = [i for i in originals["train"] if scene(i) not in test_scenes | val_scenes]
    scene_sets = {s: {scene(i) for i in kept[s]} for s in kept}
    if any(scene_sets[a] & scene_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Scene-purge failed")
    return originals, kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "mcubes_crops")
    parser.add_argument("--max-per-class", type=int, default=4)
    parser.add_argument("--purity", type=float, default=.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.purity <= 1 or args.max_per_class < 1:
        parser.error("purity must be in (0,1], max-per-class positive")
    config = {"max_per_class_per_source": args.max_per_class, "purity": args.purity,
              "seed": args.seed, "candidate_sides": list(SIZES), "minimum_validity": .99,
              "left_offset": 192, "scene_purge": True}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("config") != config or previous.get("raw_root") != str(args.raw.resolve()):
            raise ValueError("Existing crop manifest has different settings/source; choose a new output directory")
        listed = {(args.output / entry["file"]).resolve() for entry in previous["entries"]}
        actual = {path.resolve() for path in args.output.rglob("*.png")}
        if listed != actual:
            raise ValueError("Existing crop files differ from manifest; extra/missing images must be resolved")
        for entry in previous["entries"]:
            path = args.output / entry["file"]
            with Image.open(path) as image:
                digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            if digest != entry["sha256"]:
                raise ValueError(f"Previously prepared crop is missing or changed: {path}")
        print("Existing crop manifest and all image hashes verified", flush=True)
        print(json.dumps(previous["counts"], indent=2))
        return
    if args.output.exists() and any(args.output.rglob("*.png")):
        raise ValueError("Output contains image crops but has no complete manifest; choose a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    originals, kept = read_splits(args.raw)
    rng = np.random.default_rng(args.seed)
    counts, entries, seen, duplicates = {}, [], set(), 0
    for split in ("train", "val", "test"):
        counts[split] = {name: 0 for name in LABELS}
        for name in LABELS:
            (args.output / split / name).mkdir(parents=True, exist_ok=True)
        print(f"{split}: using {len(kept[split])}/{len(originals[split])} source frames", flush=True)
        for number, image_id in enumerate(kept[split], 1):
            arrays = []
            for directory in ("NIR_warped", "GT", "NIR_warped_mask"):
                path = args.raw / directory / f"{image_id}.png"
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is None or image.ndim != 2:
                    raise ValueError(f"Missing or non-grayscale source: {path}")
                arrays.append(image[:, 192:])
            nir, gt, invalid = arrays
            if nir.shape != gt.shape or gt.shape != invalid.shape:
                raise ValueError(f"Source alignment mismatch for {image_id}")
            nir = fixed_uint8(nir)
            valid = invalid == 0
            for name, label in LABELS.items():
                candidates = candidate_crops(gt, valid, label, args.max_per_class, args.purity, rng)
                for index, (box, purity, central_purity, validity, central_validity) in enumerate(candidates):
                    x, y, side = box
                    crop = Image.fromarray(nir[y:y + side, x:x + side])
                    digest = hashlib.sha256(str(crop.size).encode() + crop.tobytes()).hexdigest()
                    if digest in seen:
                        duplicates += 1
                        continue
                    seen.add(digest)
                    relative = Path(split) / name / f"{image_id}_{index}_{side}.png"
                    crop.save(args.output / relative)
                    counts[split][name] += 1
                    entries.append({"file": relative.as_posix(), "source_id": image_id,
                                    "scene": scene(image_id), "split": split, "class_name": name,
                                    "original_label_id": label, "crop_xywh_after_left_offset": [x, y, side, side],
                                    "purity": purity, "central_purity": central_purity,
                                    "validity": validity, "central_validity": central_validity,
                                    "sha256": digest})
            if number % 25 == 0 or number == len(kept[split]):
                print(f"  {number}/{len(kept[split])} sources: {counts[split]}", flush=True)
        if min(counts[split].values()) == 0:
            raise ValueError(f"At least one class has no usable crops in {split}")
    source_splits = {s: {"original": originals[s], "retained": kept[s],
                          "excluded": [i for i in originals[s] if i not in set(kept[s])],
                          "scenes": sorted({scene(i) for i in kept[s]})}
                     for s in ("train", "val", "test")}
    manifest = {"dataset": "mcubes_nir_material_crops", "source": SOURCE,
                "raw_root": str(args.raw.resolve()), "original_label_ids": LABELS,
                "config": config, "fixed_uint16_to_uint8": "round(value / 257), no per-image normalization",
                "source_splits": source_splits, "counts": counts,
                "duplicate_crops_skipped": duplicates, "inputs_are_ground_truth_masked": False,
                "entries": entries}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Prepared genuine NIR crops:", json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
