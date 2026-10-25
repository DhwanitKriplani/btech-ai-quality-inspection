"""Native-resolution semantic components exposed through the material-box API."""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from material_predictor import resolve_device
from pixel_material import CLASS_NAMES, COMPONENT_RULES, MEAN, STD, load_pixel_checkpoint, predict_material_regions


class MaterialPixelAdapter:
    names = {0: "metal", 1: "plastic", 2: "wood"}
    task = "semantic_component_boxes"
    native_resolution = True

    def __init__(self, path, device="cpu", expected_modality=None):
        self.path = Path(path).resolve()
        self.device = resolve_device(device)
        self.model, self.checkpoint = load_pixel_checkpoint(self.path, device=str(self.device))
        if (self.checkpoint.get("architecture") != "lraspp_mobilenet_v3_large"
                or self.checkpoint.get("class_names") != CLASS_NAMES):
            raise ValueError("Pixel checkpoint must be four-class LRASPP metal/plastic/wood/other")
        self.modality = self.checkpoint.get("modality")
        if self.modality not in {"rgb", "nir"} or expected_modality and self.modality != expected_modality:
            raise ValueError("Pixel checkpoint modality does not match selected sensor")
        preprocess = self.checkpoint.get("preprocess", {})
        if (preprocess.get("resize") is not None or preprocess.get("mean", MEAN) != MEAN
                or preprocess.get("std", STD) != STD or preprocess.get("color", "RGB") != "RGB"):
            raise ValueError("Pixel checkpoint must use native-resolution inference")
        self.component_rules = dict(COMPONENT_RULES, **self.checkpoint.get("component_rules", {}))
        # This selected model's component geometry was fixed before validation.
        for key in ("confidence", "minimum_area", "minimum_side", "minimum_fill", "connectivity", "max_boxes"):
            if self.component_rules[key] != COMPONENT_RULES[key]:
                raise ValueError(f"Unsupported changed pixel-component rule: {key}")
        self.effective_confidence = float(self.component_rules["confidence"])
        self.runtime_max_det = self.component_rules["max_boxes"]

    def predict(self, source, conf=.25, imgsz=None, device=None, max_det=12, verbose=False):
        """imgsz intentionally does not resize native-resolution semantic input."""
        frame = np.asarray(source)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Pixel material input must be an HxWx3 uint8 BGR frame")
        if not math.isfinite(float(conf)) or not 0 <= conf <= 1 or not isinstance(max_det, int) or max_det < 1:
            raise ValueError("Invalid pixel detector confidence/max_det")
        if device is not None and resolve_device(device) != self.device:
            raise ValueError("Pixel adapter device differs from its loaded/cached model device")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.modality == "nir" and not (np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(rgb[:, :, 1], rgb[:, :, 2])):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
        self.effective_confidence = max(float(self.component_rules["confidence"]), float(conf))
        self.runtime_max_det = min(max_det, int(self.component_rules["max_boxes"]))
        rules = dict(self.component_rules, confidence=self.effective_confidence, max_boxes=self.runtime_max_det)
        output = predict_material_regions(self.model, rgb, device=str(self.device), return_mask=True, rules=rules)
        boxes = np.asarray(output["boxes"], dtype=np.float32)
        scores = np.asarray(output["scores"], dtype=np.float32)
        classes = np.asarray(output["classes"], dtype=np.int64)
        areas = np.asarray(output["component_areas"], dtype=np.int64)
        n = len(boxes)
        if boxes.shape != (n, 4) or scores.shape != (n,) or classes.shape != (n,) or areas.shape != (n,):
            raise ValueError("Pixel model returned inconsistent component arrays")
        height, width = frame.shape[:2]
        if (not np.all(np.isfinite(boxes)) or not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1))
                or np.any((classes < 0) | (classes > 2))
                or np.any(boxes[:, :2] < 0) or np.any(boxes[:, 2] > width) or np.any(boxes[:, 3] > height)
                or np.any(boxes[:, 2:] <= boxes[:, :2])):
            raise ValueError("Pixel model returned invalid classes/coordinates")
        indices = np.flatnonzero(scores + 1e-6 >= self.effective_confidence)
        indices = indices[np.argsort(-scores[indices], kind="stable")][:self.runtime_max_det]
        pixel_mask = np.asarray(output.get("pixel_mask"))
        if pixel_mask.shape != (height, width) or np.any((pixel_mask < 0) | (pixel_mask > 3)):
            raise ValueError("Pixel mask must retain original frame dimensions and four-class labels")
        result = SimpleNamespace(
            boxes=SimpleNamespace(xyxy=torch.from_numpy(boxes[indices].copy()),
                                  conf=torch.from_numpy(scores[indices].copy()),
                                  cls=torch.from_numpy(classes[indices].astype(np.float32))),
            names=self.names, masks=None, pixel_mask=pixel_mask,
            component_areas=torch.from_numpy(areas[indices].copy()),
            backend=self.task, input_resolution="native")
        return [result]
