"""Lightweight semantic material model and fixed material-component box adapter."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights

CLASS_NAMES = ["metal", "plastic", "wood", "other"]
ORIGINAL_LABELS = {2: 0, 7: 1, 15: 2}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
COMPONENT_RULES = {"confidence": .5, "minimum_area": 256, "minimum_side": 12,
                   "minimum_fill": .10, "connectivity": 8, "max_boxes": 300,
                   "selection": "argmax equals class AND class probability >= confidence",
                   "score": "mean class probability over component pixels", "nms": False}


def map_material_labels(gt, valid):
    if gt.ndim != 2 or valid.shape != gt.shape:
        raise ValueError("Aligned two-dimensional material/validity masks required")
    target = np.full(gt.shape, 3, dtype=np.uint8)
    for old, new in ORIGINAL_LABELS.items():
        target[gt == old] = new
    target[(~valid.astype(bool)) | (gt == 255)] = 255
    return target


def build_model(pretrained=True):
    weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = lraspp_mobilenet_v3_large(weights=weights, weights_backbone=None)
    classifier = model.classifier
    classifier.low_classifier = nn.Conv2d(classifier.low_classifier.in_channels, 4, 1)
    classifier.high_classifier = nn.Conv2d(classifier.high_classifier.in_channels, 4, 1)
    return model


def normalize_image(rgb):
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("Expected HxWx3 uint8 RGB; NIR must already be repeated into RGB channels")
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float().div_(255)
    return (tensor - tensor.new_tensor(MEAN)[:, None, None]) / tensor.new_tensor(STD)[:, None, None]


def probability_components(probabilities, rules=None):
    """Return material-region boxes, not claims of complete physical objects."""
    rules = dict(COMPONENT_RULES, **(rules or {}))
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 3 or probabilities.shape[0] != 4:
        raise ValueError("Expected four class probability maps shaped 4xHxW")
    winning = probabilities.argmax(0)
    boxes = []
    for cls in range(3):
        binary = ((winning == cls) & (probabilities[cls] >= rules["confidence"])).astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for component in range(1, count):
            x, y, width, height, area = map(int, stats[component])
            if (area < rules["minimum_area"] or min(width, height) < rules["minimum_side"]
                    or area / (width * height) < rules["minimum_fill"]):
                continue
            score = float(probabilities[cls][components == component].mean())
            boxes.append((score, cls, [x, y, x + width, y + height], area))
    boxes.sort(key=lambda item: (-item[0], item[1], item[2]))
    boxes = boxes[:rules["max_boxes"]]
    return {"boxes": np.array([box[2] for box in boxes], dtype=np.float32).reshape(-1, 4),
            "scores": np.array([box[0] for box in boxes], dtype=np.float32),
            "classes": np.array([box[1] for box in boxes], dtype=np.int64),
            "component_areas": np.array([box[3] for box in boxes], dtype=np.int64)}


@torch.inference_mode()
def predict_material_regions(model, rgb, device="cpu", return_mask=False, return_probabilities=False, rules=None):
    """Native-resolution inference; no grayscale conversion or scene annotation masking."""
    tensor = normalize_image(rgb).unsqueeze(0).to(device)
    logits = model(tensor)["out"]
    probabilities = logits.float().softmax(1)[0].cpu().numpy()
    output = probability_components(probabilities, rules)
    if return_mask:
        output["pixel_mask"] = probabilities.argmax(0).astype(np.uint8)
    if return_probabilities:
        output["probabilities"] = probabilities
    return output


def load_pixel_checkpoint(path, device="cpu"):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "lraspp_mobilenet_v3_large":
        raise ValueError("Not a supported pixel-material checkpoint")
    if checkpoint.get("class_names") != CLASS_NAMES:
        raise ValueError("Unexpected pixel-material class order")
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint
