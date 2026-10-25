"""Prepare YOLO material-region boxes from MCubeS NIR or RGB and material masks.

Boxes enclose connected material regions, not necessarily separate physical objects.
Input frames remain unmasked. Test scene IDs stay out of training and validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source_tools"))
sys.path.insert(0, str(HERE.parent / "nir_material"))
from prepare_mcubes_crops import LABELS, SOURCE, fixed_uint8, read_splits, scene

CLASS_NAMES = list(LABELS)
LEFT_OFFSET = 192


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_rgb_uint8(image):
    """Preserve polL_color BGR channels with the same fixed intensity scale."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected three-channel polL_color RGB camera image")
    if image.dtype == np.uint16:
        return np.rint(image.astype(np.float32) / 257.0).astype(np.uint8)
    if image.dtype == np.uint8:
        return image
    raise ValueError(f"Unsupported RGB dtype {image.dtype}")


def region_boxes(gt, valid, minimum_area=256, minimum_side=12, minimum_fill=0.10):
    """8-connected valid material components in already left-cropped coordinates."""
    if gt.ndim != 2 or gt.shape != valid.shape:
        raise ValueError("Aligned two-dimensional material and validity masks required")
    height, width = gt.shape
    boxes = []
    for class_id, name in enumerate(CLASS_NAMES):
        binary = ((gt == LABELS[name]) & valid).astype(np.uint8)
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for index in range(1, component_count):
            x, y, w, h, area = map(int, stats[index])
            fill = area / (w * h)
            if area < minimum_area or min(w, h) < minimum_side or fill < minimum_fill:
                continue
            normalized = [(x + w / 2) / width, (y + h / 2) / height, w / width, h / height]
            if not all(0 <= value <= 1 for value in normalized) or min(normalized[2:]) <= 0:
                raise ValueError("Invalid normalized component box")
            boxes.append({"class_id": class_id, "class_name": name,
                          "original_label_id": LABELS[name],
                          "xywh_after_left_offset": [x, y, w, h],
                          "xyxy_original": [x + LEFT_OFFSET, y, x + w + LEFT_OFFSET, y + h],
                          "normalized_xywh": normalized, "valid_material_pixels": area,
                          "bbox_fill": fill})
    return boxes


def yaml_text(output):
    return (f"path: {json.dumps(output.resolve().as_posix())}\n"
            "train: train/images\nval: val/images\ntest: test/images\n"
            f"names: {json.dumps(CLASS_NAMES)}\n")


def verify_prepared(output, raw, config, kept):
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous_config = dict(manifest.get("config", {}))
    legacy_nir = ("modality" not in previous_config and
                  manifest.get("dataset") == "mcubes_nir_material_region_detection")
    if legacy_nir:
        previous_config["modality"] = "nir"
    if previous_config != config or manifest.get("raw_root") != str(raw):
        raise ValueError("Existing manifest uses different source/settings; choose a new output folder")
    if {s: manifest["source_splits"][s]["retained"] for s in kept} != kept:
        raise ValueError("Official scene-purged split IDs changed; choose a new output folder")
    expected = {manifest_path.resolve(), (output / "dataset.yaml").resolve()}
    for entry in manifest["entries"]:
        for key, hash_key in (("image", "image_sha256"), ("label", "label_sha256")):
            path = (output / entry[key]).resolve()
            if not path.is_relative_to(output.resolve()):
                raise ValueError("Manifest references a file outside output")
            expected.add(path)
            if not path.is_file() or sha256(path) != entry[hash_key]:
                raise ValueError(f"Prepared {key} is missing or changed: {path}")
        for directory, digest in entry["source_hashes"].items():
            source_path = raw / directory / f"{entry['source_id']}.png"
            if not source_path.is_file() or sha256(source_path) != digest:
                raise ValueError(f"Source file is missing or changed: {source_path}")
    actual = {p.resolve() for p in output.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("Prepared folder has extra/missing files; resolve them or choose a new output folder")
    if (output / "dataset.yaml").read_text(encoding="utf-8") != yaml_text(output):
        raise ValueError("dataset.yaml changed or refers to another output folder")
    if legacy_nir:
        # Earlier NIR-only output is unambiguous; upgrade provenance only after
        # checking every source, image, label, YAML and retained source ID.
        manifest["modality"] = "nir"
        manifest["config"] = previous_config
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Existing detection dataset: all output/source hashes and split IDs verified.", flush=True)
    print(json.dumps(manifest["counts"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--modality", choices=("nir", "rgb"), default="nir")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-area", type=int, default=256)
    parser.add_argument("--min-side", type=int, default=12)
    parser.add_argument("--min-fill", type=float, default=0.10)
    args = parser.parse_args()
    if min(args.min_area, args.min_side) < 1 or not 0 < args.min_fill <= 1:
        parser.error("min-area/min-side must be positive and min-fill must be in (0,1]")
    default_output = HERE / ("material_detection_data" if args.modality == "nir" else "material_rgb_detection_data")
    raw, output = args.raw.resolve(), (args.output or default_output).resolve()
    config = {"modality": args.modality, "left_offset": LEFT_OFFSET, "connectivity": 8,
              "minimum_component_area": args.min_area, "minimum_bbox_side": args.min_side,
              "minimum_bbox_fill": args.min_fill, "scene_purge": True,
              "invalid_mask_rule": "NIR_warped_mask == 0 is valid",
              "intensity_conversion": "uint16 -> round(value/257); uint8 unchanged",
              "input_channels": ("same NIR intensity repeated into RGB; no GT masking"
                                 if args.modality == "nir" else "polL_color RGB color; no GT masking")}
    originals, kept = read_splits(raw)
    if (output / "manifest.json").is_file():
        verify_prepared(output, raw, config, kept)
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is nonempty without a complete manifest; choose a new output folder")
    output.mkdir(parents=True, exist_ok=True)
    entries, counts = [], {}
    for split in ("train", "val", "test"):
        image_dir, label_dir = output / split / "images", output / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        counts[split] = {"images": 0, "negative_images": 0,
                         "boxes": {name: 0 for name in CLASS_NAMES}}
        print(f"{split}: {len(kept[split])}/{len(originals[split])} scene-purged source frames", flush=True)
        for number, image_id in enumerate(kept[split], 1):
            arrays, source_hashes = [], {}
            frame_directory = "NIR_warped" if args.modality == "nir" else "polL_color"
            for directory in (frame_directory, "GT", "NIR_warped_mask"):
                path = raw / directory / f"{image_id}.png"
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                expected_dims = 3 if directory == "polL_color" else 2
                if (image is None or image.ndim != expected_dims or image.shape[1] <= LEFT_OFFSET
                        or (expected_dims == 3 and image.shape[2] != 3)):
                    raise ValueError(f"Missing/wrong-channel/too-small source image: {path}")
                arrays.append(image[:, LEFT_OFFSET:])
                source_hashes[directory] = sha256(path)
            frame, gt, invalid = arrays
            if frame.shape[:2] != gt.shape or gt.shape != invalid.shape:
                raise ValueError(f"Source alignment mismatch: {image_id}")
            source_dtype = str(frame.dtype)
            if args.modality == "nir":
                image8 = fixed_uint8(frame)
                image_rgb = np.repeat(image8[:, :, None], 3, axis=2)
            else:
                image_rgb = cv2.cvtColor(fixed_rgb_uint8(frame), cv2.COLOR_BGR2RGB)
            boxes = region_boxes(gt, invalid == 0, args.min_area, args.min_side, args.min_fill)
            image_relative = Path(split) / "images" / f"{image_id}.png"
            label_relative = Path(split) / "labels" / f"{image_id}.txt"
            image_path, label_path = output / image_relative, output / label_relative
            Image.fromarray(image_rgb).save(image_path)
            labels = "".join(f"{box['class_id']} " + " ".join(f"{v:.9f}" for v in box["normalized_xywh"]) + "\n"
                             for box in boxes)
            label_path.write_text(labels, encoding="utf-8")
            counts[split]["images"] += 1
            counts[split]["negative_images"] += not boxes
            for box in boxes:
                counts[split]["boxes"][box["class_name"]] += 1
            entries.append({"source_id": image_id, "scene": scene(image_id), "split": split,
                            "image": image_relative.as_posix(), "label": label_relative.as_posix(),
                            "width": int(image_rgb.shape[1]), "height": int(image_rgb.shape[0]),
                            "source_dtype": source_dtype, "box_count": len(boxes), "boxes": boxes,
                            "source_hashes": source_hashes,
                            "image_sha256": sha256(image_path), "label_sha256": sha256(label_path)})
            if number % 25 == 0 or number == len(kept[split]):
                print(f"  {number}/{len(kept[split])}: {counts[split]}", flush=True)
    source_splits = {
        split: {"original": originals[split], "retained": kept[split],
                "excluded": [i for i in originals[split] if i not in set(kept[split])],
                "scenes": sorted({scene(i) for i in kept[split]})}
        for split in ("train", "val", "test")
    }
    manifest = {"schema_version": 1, "dataset": f"mcubes_{args.modality}_material_region_detection",
                "modality": args.modality,
                "dataset_source": SOURCE, "raw_root": str(raw),
                "sensor_name": ("FLIR GS3-U3-41C6NIR-C; 750 nm long-pass" if args.modality == "nir"
                                else "MCubeS polL_color RGB camera channel"),
                "label_semantics": "connected material regions, not guaranteed physical object instances",
                "class_names": CLASS_NAMES, "original_label_ids": LABELS,
                "config": config, "inputs_are_ground_truth_masked": False,
                "source_splits": source_splits, "counts": counts, "entries": entries}
    (output / "dataset.yaml").write_text(yaml_text(output), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Prepared detection dataset:", json.dumps(counts, indent=2), flush=True)
    print(f"YOLO dataset: {output / 'dataset.yaml'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
