"""Run material boxes/classification and aluminium surface-defect inspection together."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageOps
from calibrate_views import LIMITATION as CALIBRATION_LIMITATION, load_calibration, map_points, verify_dimensions

HERE = Path(__file__).resolve().parent
MATERIAL_NAMES = ("metal", "plastic", "wood")
DEFECT_NAMES = ("Black Marker", "Corner Defect", "Side Defect", "scratch")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
HEADER_HEIGHT = 42


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def names_list(names):
    return [str(names[k]) for k in sorted(names)] if isinstance(names, dict) else list(names)


def clip_box(box, width, height):
    """Clamp a predicted box without inventing a replacement for invalid boxes."""
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        return None
    x1, y1, x2, y2 = map(float, box)
    x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
    y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def associate_region(defect_box, materials):
    """Same-frame geometric association only; not cross-sensor registration."""
    cx = (defect_box[0] + defect_box[2]) / 2
    cy = (defect_box[1] + defect_box[3]) / 2
    containing = [m for m in materials if m["bbox_xyxy"][0] <= cx <= m["bbox_xyxy"][2]
                  and m["bbox_xyxy"][1] <= cy <= m["bbox_xyxy"][3]]
    if not containing:
        return None
    smallest = min(containing, key=lambda m: (m["bbox_xyxy"][2] - m["bbox_xyxy"][0])
                   * (m["bbox_xyxy"][3] - m["bbox_xyxy"][1]))
    return smallest["region_id"]


def safe_frame(frame):
    if frame is None or frame.dtype != np.uint8:
        raise ValueError("Expected normalized 8-bit camera/image pixels; raw/16-bit inputs need fixed sensor normalization")
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a grayscale or three-channel image")
    return frame


def read_photo(path):
    path = Path(path)
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Use a single photo or video input, not a multi-frame image")
        high_depth = image.mode in {"I", "F"} or image.mode.startswith("I;")
        if image.format == "PNG":
            with path.open("rb") as stream:
                header = stream.read(25)
            high_depth |= len(header) == 25 and header[24] > 8
        if hasattr(image, "tag_v2"):
            bits = image.tag_v2.get(258, (8,))
            bits = (bits,) if isinstance(bits, int) else bits
            high_depth |= any(x > 8 for x in bits)
        if high_depth:
            raise ValueError("16-bit/float/raw photos need fixed sensor normalization before inspection")
        pixels = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)


def draw_text(frame, text, point, color=(255, 255, 255), size=.47):
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 1)
    point = (max(0, min(int(point[0]), frame.shape[1] - text_width - 3)),
             max(text_height + 1, min(int(point[1]), frame.shape[0] - baseline - 2)))
    cv2.putText(frame, text, point, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, point, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


class InspectionPipeline:
    def __init__(self, model_dir=HERE / "models", modalities=("rgb",), device="auto", imgsz=640,
                 material_conf=.25, defect_conf=.35, max_materials=12, threads=4,
                 include_markers=False, material_imgsz=None, defect_imgsz=None, backgrounds=None, background_threshold=25., foreground_min_area=.002):
        import torch
        from ultralytics import YOLO
        from material_predictor import MaterialPredictor

        torch.set_num_threads(threads)
        if device == "auto":
            device = "0" if torch.cuda.is_available() else "cpu"
        if device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment; use --device cpu or the configured GPU environment")
        self.device, self.imgsz = device, imgsz
        self.material_imgsz = imgsz if material_imgsz is None else material_imgsz
        self.defect_imgsz = imgsz if defect_imgsz is None else defect_imgsz
        if any(not isinstance(size, int) or size < 1 for size in (self.material_imgsz, self.defect_imgsz)):
            raise ValueError("Material and defect input sizes must be positive integers")
        self.threads = threads
        self.material_conf, self.defect_conf = material_conf, defect_conf
        self.max_materials, self.include_markers = max_materials, include_markers
        self.material_detectors, self.classifiers, self.model_metadata = {}, {}, {}
        self.defect_detector = None
        self.backgrounds = backgrounds or {}
        root = Path(model_dir).resolve()

        def required(name):
            path = root / name
            if not path.is_file():
                raise FileNotFoundError(f"Required trained model is missing: {path}")
            return path

        def describe(role, path, model, domain):
            self.model_metadata[role] = {"path": str(path), "sha256": file_hash(path), "domain": domain,
                                         "names": names_list(model.names), "task": model.task}

        for modality in set(modalities):
            if modality not in {"rgb", "nir"}:
                raise ValueError(f"Unsupported modality: {modality}")
            pixel_path = root / f"material_{modality}_pixel.pt"
            if modality in self.backgrounds:
                from background_localizer import BackgroundLocalizer
                path = Path(self.backgrounds[modality]).resolve()
                detector = BackgroundLocalizer(read_photo(path), modality, background_threshold, foreground_min_area)
            elif pixel_path.is_file():
                from material_pixel_adapter import MaterialPixelAdapter
                path = pixel_path
                detector = MaterialPixelAdapter(path, device=self.device, expected_modality=modality)
            else:
                path = required(f"material_{modality}_detector.pt")
                detector = YOLO(str(path))
            if names_list(detector.names) != list(MATERIAL_NAMES):
                raise ValueError(f"Material detector must be custom-trained with class order {MATERIAL_NAMES}: {path}")
            self.material_detectors[modality] = detector
            describe(f"material_{modality}_detector", path, detector,
                     "Observed empty-reference foreground geometry; classification is separate" if getattr(detector,"task","")=="background_difference" else f"MCubeS {modality.upper()} material regions")
            self.model_metadata[f"material_{modality}_detector"].update(
                input_resolution="native" if getattr(detector, "native_resolution", False) is True else self.material_imgsz,
                component_rules=getattr(detector, "component_rules", None))
            path = required(f"material_{modality}_classifier.pt")
            classifier = MaterialPredictor(path, threads=threads, device=self.device)
            if list(classifier.class_names) != list(MATERIAL_NAMES):
                raise ValueError(f"Unexpected classifier class order: {path}")
            self.classifiers[modality] = classifier
            self.model_metadata[f"material_{modality}_classifier"] = {
                "path": str(path), "sha256": file_hash(path), "domain": classifier.training_domain,
                "names": list(classifier.class_names), "task": "classify", "device": str(classifier.device),
                "batch_inference": True, "ensemble": classifier.ensemble}
        if "rgb" in modalities:
            path = required("defect_detector.pt")
            self.defect_detector = YOLO(str(path))
            if names_list(self.defect_detector.names) != list(DEFECT_NAMES):
                raise ValueError(f"Expected the trained aluminium-defect class order {DEFECT_NAMES}: {path}")
            describe("defect_detector", path, self.defect_detector, "RGB aluminium surface defects; other materials unvalidated")

    def metadata(self):
        material_sizes = {name: "native" if getattr(model, "native_resolution", False) is True else self.material_imgsz
                          for name, model in self.material_detectors.items()}
        effective_size = next(iter(material_sizes.values())) if len(set(material_sizes.values())) == 1 else material_sizes
        return {"models": self.model_metadata, "detector_device": self.device,
                "imgsz": self.imgsz if effective_size == self.defect_imgsz == self.imgsz else None,
                "material_imgsz": effective_size, "material_configured_yolo_imgsz": self.material_imgsz,
                "material_input_size_by_sensor": material_sizes, "defect_imgsz": self.defect_imgsz,
                "threads": self.threads, "max_material_regions": self.max_materials,
                "material_conf": self.material_conf, "defect_conf": self.defect_conf,
                "markers_included": self.include_markers,
                "limitations": [
                    "Learned boxes are material regions; foreground mode requires an empty reference, fixed camera and unchanged lighting.",
                    "Detector labels and crop-classifier labels are separate estimates; scores are uncalibrated.",
                    "The surface-defect model was trained on RGB aluminium; wood/plastic/NIR defect performance is unvalidated.",
                    "No detection means no defect detected, not a guarantee of a defect-free part.",
                    "Separate RGB/NIR sources are not registered or hardware-synchronized; boxes are not transferred across sensors."]}

    def process_frame(self, frame, modality="rgb"):
        frame = safe_frame(frame)
        height, width = frame.shape[:2]
        started = time.perf_counter()
        if modality not in self.material_detectors:
            raise ValueError(f"Models for {modality} were not loaded")
        result = self.material_detectors[modality].predict(
            source=frame, conf=self.material_conf, imgsz=getattr(self, "material_imgsz", self.imgsz), device=self.device,
            max_det=self.max_materials, verbose=False)[0]
        is_foreground = getattr(self.material_detectors[modality], "task", "") == "background_difference"
        materials, crops = [], []
        if result.boxes is not None:
            component_areas = getattr(result, "component_areas", None)
            area_values = component_areas.cpu().tolist() if isinstance(component_areas, np.ndarray) is False and hasattr(component_areas, "cpu") else None
            for index, (raw, score, label) in enumerate(zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                                         result.boxes.cls.cpu().tolist())):
                box = clip_box(raw, width, height)
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                crop = frame[math.floor(y1):math.ceil(y2), math.floor(x1):math.ceil(x2)]
                if crop.size == 0:
                    continue
                detector_label = None if is_foreground else str(result.names[int(label)])
                crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                materials.append({"region_id": f"M{len(materials) + 1}", "region_kind": "foreground_object" if is_foreground else "material_region",
                                  "bbox_xyxy": box, "detector_class_id": None if is_foreground else int(label),
                                  "detector_label": detector_label, "detector_score": None if is_foreground else float(score),
                                  "localization_method": "background_difference" if is_foreground else "learned_material_region"})
                if isinstance(area_values, list) and index < len(area_values):
                    materials[-1]["foreground_area_pixels" if is_foreground else "semantic_component_area_pixels"] = int(area_values[index])
        if crops:
            predictor = self.classifiers[modality]
            # Older lightweight adapters may expose only scalar predict(); do not invent crops.
            if callable(getattr(type(predictor), "predict_many", None)):
                classifications = predictor.predict_many(crops)
            else:
                classifications = [predictor.predict(crop) for crop in crops]
            if not isinstance(classifications, list) or len(classifications) != len(materials):
                raise ValueError("Crop classifier returned an invalid result count")
            for item, classification in zip(materials, classifications):
                item["classifier"] = classification
                item["classifier_agrees"] = None if is_foreground else classification["prediction"] == item["detector_label"]
                if is_foreground: item["material_estimate"] = classification["prediction"]
        defects = {"status": "not_run_wrong_modality", "items": [], "surface_defect_count": 0,
                   "marker_count": 0, "training_domain": "RGB aluminium"}
        masks = []
        if modality == "rgb":
            classes = [0, 1, 2, 3] if self.include_markers else [1, 2, 3]
            result = self.defect_detector.predict(source=frame, conf=self.defect_conf, imgsz=getattr(self, "defect_imgsz", self.imgsz),
                                                 device=self.device, classes=classes, verbose=False)[0]
            defects["status"] = "ran"
            if result.boxes is not None:
                polygons = result.masks.xy if result.masks is not None else []
                for index, (raw, score, label) in enumerate(zip(result.boxes.xyxy.cpu().tolist(),
                        result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist())):
                    class_id = int(label)
                    if class_id not in classes:
                        continue
                    box = clip_box(raw, width, height)
                    if box is None:
                        continue
                    if is_foreground and associate_region(box, materials) is None:
                        continue
                    item = {"detection_id": f"D{len(defects['items']) + 1}", "bbox_xyxy": box,
                            "class_id": class_id, "label": str(result.names[class_id]), "score": float(score),
                            "kind": "marker" if class_id == 0 else "surface_defect",
                            "material_region_id": associate_region(box, materials),
                            "association_method": "smallest_same_RGB_region_containing_defect_center",
                            "validated_for_material": "aluminium training domain only"}
                    if index < len(polygons) and len(polygons[index]) >= 3:
                        poly = np.asarray(polygons[index], dtype=np.float32)
                        poly[:, 0] = np.clip(poly[:, 0], 0, width - 1)
                        poly[:, 1] = np.clip(poly[:, 1], 0, height - 1)
                        simplified = cv2.approxPolyDP(poly.reshape(-1, 1, 2), 1.0, True).reshape(-1, 2)
                        if len(simplified) >= 3:
                            poly = simplified
                        item["mask_polygon_xy"] = np.round(poly, 2).tolist()
                        masks.append((poly.astype(np.int32), class_id))
                    defects["items"].append(item)
            defects["surface_defect_count"] = sum(x["kind"] == "surface_defect" for x in defects["items"])
            defects["marker_count"] = sum(x["kind"] == "marker" for x in defects["items"])
        output = frame.copy()
        overlay = output.copy()
        for polygon, class_id in masks:
            cv2.fillPoly(overlay, [polygon], (180, 80, 255) if class_id == 0 else (30, 30, 255))
        if masks:
            output = cv2.addWeighted(overlay, .25, output, .75, 0)
        for item in materials:
            x1, y1, x2, y2 = [round(x) for x in item["bbox_xyxy"]]
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 220), 2)
            first_line = y1 - 23 if y1 >= 40 else y1 + 15
            second_line = first_line + 18
            primary = (f"{item['region_id']} estimate: {item['classifier']['prediction']} {item['classifier']['score']:.2f}"
                       if is_foreground else f"{item['region_id']} {item['detector_label']} {item['detector_score']:.2f}")
            draw_text(output, primary, (x1 + 2, first_line), (0, 255, 255))
            material_check = (f"check: {item['classifier']['prediction']} {item['classifier']['score']:.2f}"
                              if item["classifier_agrees"] else "Foreground object" if is_foreground else "Material check: unsure")
            draw_text(output, material_check,
                      (x1 + 2, second_line), (0, 255, 255))
        for item in defects["items"]:
            x1, y1, x2, y2 = [round(x) for x in item["bbox_xyxy"]]
            color = (220, 80, 255) if item["kind"] == "marker" else (50, 50, 255)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            draw_text(output, f"{item['label']} {item['score']:.2f}", (x1 + 2, max(15, y1 - 5)), color)
        elapsed = (time.perf_counter() - started) * 1000
        canvas = np.full((height + HEADER_HEIGHT, width, 3), 25, dtype=np.uint8)
        canvas[HEADER_HEIGHT:] = output
        title = f"{modality.upper()} | regions {len(materials)}"
        title += f" | defects {defects['surface_defect_count']} (aluminium model)" if modality == "rgb" else " | RGB defects inactive"
        draw_text(canvas, title, (8, 17), size=.45)
        draw_text(canvas, f"{elapsed:.0f} ms | scores uncalibrated", (8, 35), size=.4)
        report = {"modality": modality, "original_size_wh": [width, height], "coordinate_system": "original_source_pixels",
                  "material_detector_backend": str(getattr(self.material_detectors[modality], "task", "detect")),
                  "material_input_resolution": "native" if getattr(self.material_detectors[modality], "native_resolution", False) is True else getattr(self, "material_imgsz", self.imgsz),
                  "material_effective_component_confidence": getattr(self.material_detectors[modality], "effective_confidence", None),
                  "materials": materials, "defects": defects, "inference_ms": elapsed,
                  "inspection_status": "no_object_detected" if is_foreground and not materials else "no_defect_detected" if modality == "rgb" and not defects["surface_defect_count"]
                                       else "detections_present" if modality == "rgb" else "material_only"}
        return canvas, report


def compose_panels(panels):
    """Return display geometry while preserving each panel's source coordinates."""
    if len(panels) == 1:
        name, image, report = panels[0]
        report["display_transform"] = {"scale": 1.0, "offset_x": 0, "offset_y": HEADER_HEIGHT}
        return image, {name: report}
    target_height = min(720, max(image.shape[0] for _, image, _ in panels))
    rendered, reports, offset = [], {}, 0
    for name, image, report in panels:
        scale = target_height / image.shape[0]
        width = max(1, round(image.shape[1] * scale))
        rendered.append(cv2.resize(image, (width, target_height)))
        report["display_transform"] = {"scale_x": width / image.shape[1], "scale_y": scale,
                                       "offset_x": offset, "offset_y": HEADER_HEIGHT * scale}
        reports[name] = report
        offset += width
    return np.concatenate(rendered, axis=1), reports


def calibrated_associations(panels, calibration):
    """Project RGB detections on the calibrated plane; keep original sensor coordinates."""
    by_name = {name: (image, report) for name, image, report in panels}
    if set(by_name) != {"rgb", "nir"}:
        raise ValueError("Calibration requires one RGB and one NIR panel")
    nir_image, nir = by_name["nir"]
    rgb = by_name["rgb"][1]
    verify_dimensions(calibration, rgb["original_size_wh"], nir["original_size_wh"])
    width, height = nir["original_size_wh"]
    associations = []
    for defect in rgb["defects"]["items"]:
        x1, y1, x2, y2 = defect["bbox_xyxy"]
        center = [[(x1 + x2) / 2, (y1 + y2) / 2]]
        corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        association = {"rgb_detection_id": defect["detection_id"], "label": defect["label"],
                       "rgb_material_region_id": defect.get("material_region_id"),
                       "nir_material_region_id": None,
                       "method": "calibrated_RGB_center_to_smallest_containing_NIR_region",
                       "mapped_shape_kind": "projected_quadrilateral_not_axis_aligned_box",
                       "score_fusion": "none", "calibration_limitation": CALIBRATION_LIMITATION}
        try:
            mapped_center = map_points(center, calibration["matrix"])[0]
            mapped_corners = map_points(corners, calibration["matrix"])
            association.update(mapped_center_xy=mapped_center.tolist(),
                               mapped_box_polygon_xy=mapped_corners.tolist())
            polygon = mapped_corners
            if defect.get("mask_polygon_xy"):
                polygon = map_points(defect["mask_polygon_xy"], calibration["matrix"])
                association["mapped_mask_polygon_xy"] = polygon.tolist()
            inside = 0 <= mapped_center[0] < width and 0 <= mapped_center[1] < height
            association["status"] = "mapped" if inside else "outside_nir_frame"
            if inside:
                cx, cy = mapped_center
                association["nir_material_region_id"] = associate_region([cx, cy, cx, cy], nir["materials"])
                drawable = np.rint(polygon).astype(np.int32)
                drawable[:, 1] += HEADER_HEIGHT
                cv2.polylines(nir_image, [drawable.reshape(-1, 1, 2)], True, (255, 0, 255), 2, cv2.LINE_AA)
                draw_text(nir_image, f"RGB {defect['label']} mapped", (round(cx), round(cy) + HEADER_HEIGHT - 6),
                          (255, 80, 255), .42)
        except ValueError as exc:
            association.update(status="projection_failed", error=str(exc))
        defect["cross_sensor_association"] = association
        associations.append(association)
    nir["mapped_rgb_detection_count"] = sum(item["status"] == "mapped" for item in associations)
    nir_image[:HEADER_HEIGHT] = 25
    draw_text(nir_image, f"NIR | regions {len(nir['materials'])} | mapped RGB detections {nir['mapped_rgb_detection_count']}", (8, 17), size=.43)
    draw_text(nir_image, "Planar calibration; capture is not hardware-synchronized", (8, 35), size=.38)
    return associations


class SourceReader:
    def __init__(self, value):
        self.value = str(value)
        self.capture = None
        self.image = None
        self.image_used = False
        self.kind = "picamera2" if self.value.startswith("picamera2:") else "camera" if self.value.isdigit() else "image" if Path(self.value).suffix.lower() in IMAGE_SUFFIXES else "video"
        self.fps = 20.0

    def open(self):
        if self.kind == "picamera2":
            try:
                from picamera2 import Picamera2
            except ImportError as exc:
                raise RuntimeError("CSI camera needs system python3-picamera2 and a --system-site-packages environment") from exc
            index_text = self.value.split(":",1)[1]
            if not index_text.isdigit(): raise ValueError("CSI source must be picamera2:0 or another nonnegative camera index")
            self.capture = Picamera2(int(index_text))
            config = self.capture.create_video_configuration(main={"size":(640,480),"format":"RGB888"},controls={"FrameRate":15},buffer_count=2)
            self.capture.configure(config)
            self.capture.start()
            self.fps = 15.
        elif self.kind == "image":
            self.image = read_photo(self.value)
        else:
            if self.kind == "video" and not Path(self.value).is_file():
                raise FileNotFoundError(f"Video not found: {self.value}")
            self.capture = cv2.VideoCapture(int(self.value) if self.kind == "camera" else self.value)
            if not self.capture.isOpened():
                raise RuntimeError(f"Cannot open {self.kind} source: {self.value}")
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            if math.isfinite(fps) and .1 < fps < 300:
                self.fps = fps
        return self

    def read(self):
        if self.kind == "picamera2":
            # Picamera2 RGB888 is used directly as BGR by the official Ultralytics Pi example.
            return True, np.ascontiguousarray(self.capture.capture_array("main")[:,:,:3])
        if self.kind == "image":
            if self.image_used:
                return False, None
            self.image_used = True
            return True, self.image.copy()
        return self.capture.read()

    def close(self):
        if self.capture is not None:
            if self.kind == "picamera2":
                try: self.capture.stop()
                finally: self.capture.close()
            else:
                self.capture.release()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="RGB photo/video path or explicit Windows camera index")
    parser.add_argument("--input-mode", choices=("rgb", "nir"), default="rgb")
    parser.add_argument("--nir-source", help="Additional NIR photo/video/camera; unregistered side-by-side output")
    parser.add_argument("--calibration", type=Path, help="Optional validated RGB-to-NIR planar calibration JSON; requires --nir-source")
    parser.add_argument("--model-dir", type=Path, default=HERE / "models")
    parser.add_argument("--output", type=Path, default=HERE / "outputs", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Explicitly open a live result window")
    parser.add_argument("--device", default="auto", help="auto, cpu, or CUDA device index such as 0")
    parser.add_argument("--imgsz", type=int, default=None, help="Legacy common override: explicitly sets both detector input sizes")
    parser.add_argument("--material-imgsz", type=int, default=640, help="Retained material detector input size (default640)")
    parser.add_argument("--defect-imgsz", type=int, default=640, help="RGB defect detector input size (default640)")
    parser.add_argument("--material-conf", type=float, default=.25)
    parser.add_argument("--defect-conf", type=float, default=.35)
    parser.add_argument("--max-materials", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth video/camera frame")
    parser.add_argument("--max-frames", type=int, help="Optional processed-frame limit")
    parser.add_argument("--include-markers", action="store_true", help="Also display Black Marker, separately from defects")
    parser.add_argument("--background-rgb", type=Path, help="Empty fixed-camera RGB reference for foreground object localization")
    parser.add_argument("--background-nir", type=Path, help="Empty fixed-camera NIR reference for foreground object localization")
    parser.add_argument("--background-threshold", type=float, default=25.)
    parser.add_argument("--foreground-min-area", type=float, default=.002)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.nir_source and args.input_mode != "rgb":
        parser.error("--nir-source requires a main RGB source (--input-mode rgb)")
    if args.calibration and not args.nir_source:
        parser.error("--calibration requires --nir-source with a main RGB input")
    calibration = load_calibration(args.calibration) if args.calibration else None
    sizes = [args.material_imgsz, args.defect_imgsz] + ([args.imgsz] if args.imgsz is not None else [])
    if min(*sizes, args.max_materials, args.threads, args.stride) < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("Detector sizes, max-materials, threads, stride and max-frames must be positive")
    if not 0 <= args.material_conf <= 1 or not 0 <= args.defect_conf <= 1:
        parser.error("Confidence thresholds must be between 0 and 1")
    if args.device not in {"auto", "cpu"} and not args.device.isdigit():
        parser.error("device must be auto, cpu or a nonnegative CUDA device index")
    sources = [(args.input_mode, SourceReader(args.source))]
    if args.nir_source:
        sources.append(("nir", SourceReader(args.nir_source)))
        if (sources[0][1].kind == "image") != (sources[1][1].kind == "image"):
            parser.error("For paired input, supply two photos or two video/camera streams")
        if sources[0][1].kind in {"camera","picamera2"} and sources[1][1].kind in {"camera","picamera2"} and args.source == args.nir_source:
            parser.error("Paired cameras must use different device indices")
    used_modalities = {name for name,_ in sources}
    if args.background_rgb and "rgb" not in used_modalities or args.background_nir and "nir" not in used_modalities:
        parser.error("Background reference supplied for a sensor that is not active")
    material_size = args.imgsz if args.imgsz is not None else args.material_imgsz
    defect_size = args.imgsz if args.imgsz is not None else args.defect_imgsz
    pipeline = InspectionPipeline(args.model_dir, [name for name, _ in sources], args.device, 640 if args.imgsz is None else args.imgsz,
                                  args.material_conf, args.defect_conf, args.max_materials,
                                  args.threads, args.include_markers,
                                  material_imgsz=material_size, defect_imgsz=defect_size,
                                  backgrounds={k:v for k,v in {"rgb":args.background_rgb,"nir":args.background_nir}.items() if v},
                                  background_threshold=args.background_threshold, foreground_min_area=args.foreground_min_area)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = "inspection_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    is_photo = sources[0][1].kind == "image"
    media_path = args.output / (stem + (".jpg" if is_photo else ".mp4"))
    records_path = args.output / (stem + (".json" if is_photo else ".jsonl"))
    summary_path = args.output / (stem + ".summary.json")
    run = {"created_utc": datetime.now(timezone.utc).isoformat(), "input_mode": args.input_mode,
           "sources": {name: {"source": reader.value, "kind": reader.kind} for name, reader in sources},
           "sensor_registration": "not_registered" if len(sources) > 1 else "single_source",
           "capture_synchronization": "sequential_reads_not_hardware_synchronized" if len(sources) > 1 else "single_source",
           "cross_sensor_associations": None, "pipeline": pipeline.metadata(),
           "output_media": str(media_path.resolve()), "output_records": str(records_path.resolve())}
    if calibration:
        run["sensor_registration"] = "planar_homography_rgb_to_nir"
        run["cross_sensor_associations"] = {"status": "enabled", "stored_in": "per-frame records"}
        run["calibration"] = {"path": str(args.calibration.resolve()), "sha256": file_hash(args.calibration),
                              "direction": calibration["direction"], "matrix": calibration["matrix"],
                              "rgb_size_wh": calibration["rgb_size_wh"], "nir_size_wh": calibration["nir_size_wh"],
                              "reprojection": calibration["reprojection"], "limitation": CALIBRATION_LIMITATION}
        run["pipeline"]["limitations"] = [text for text in run["pipeline"].get("limitations", [])
                                             if not text.startswith("Separate RGB/NIR sources are not registered")]
        run["pipeline"]["limitations"].append(CALIBRATION_LIMITATION)
    writer, records, processed, frame_index = None, None, 0, 0
    interrupted = False
    try:
        for _, reader in sources:
            reader.open()
        if not is_photo:
            records = records_path.open("w", encoding="utf-8")
        while True:
            frames = [(name, reader.read()) for name, reader in sources]
            if not all(ok for _, (ok, _) in frames):
                if processed == 0:
                    raise RuntimeError("No readable frames from one or more sources")
                break
            if calibration:
                by_name = {name: frame for name, (_, frame) in frames}
                verify_dimensions(calibration, [by_name["rgb"].shape[1], by_name["rgb"].shape[0]],
                                  [by_name["nir"].shape[1], by_name["nir"].shape[0]])
            current_index = frame_index
            captured_utc = datetime.now(timezone.utc).isoformat()
            frame_index += 1
            if not is_photo and current_index % args.stride:
                continue
            panels = []
            for name, (_, frame) in frames:
                annotated, report = pipeline.process_frame(frame, name)
                panels.append((name, annotated, report))
            cross_associations = calibrated_associations(panels, calibration) if calibration else None
            combined, reports = compose_panels(panels)
            record = {"processed_frame": processed, "source_frame_index": current_index,
                      "captured_utc": captured_utc, "panels": reports,
                      "cross_sensor_associations": cross_associations}
            if is_photo:
                if not cv2.imwrite(str(media_path), combined):
                    raise RuntimeError(f"Cannot save result image: {media_path}")
                records_path.write_text(json.dumps(dict(run, frame=record), indent=2), encoding="utf-8")
            else:
                if writer is None:
                    writer = cv2.VideoWriter(str(media_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             sources[0][1].fps / args.stride, (combined.shape[1], combined.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create output video: {media_path}")
                writer.write(combined)
                records.write(json.dumps(record) + "\n")
                records.flush()
            processed += 1
            for name, report in reports.items():
                print(f"Frame {current_index} {name.upper()}: material regions={len(report['materials'])}, "
                      f"surface defects={report['defects']['surface_defect_count']}, "
                      f"markers={report['defects']['marker_count']}", flush=True)
            if args.show:
                cv2.imshow("Integrated inspection - Q or Escape to stop", combined)
                key = cv2.waitKey(1) & 0xFF
                if is_photo:
                    while key not in (27, ord("q"), ord("Q")):
                        if cv2.getWindowProperty("Integrated inspection - Q or Escape to stop", cv2.WND_PROP_VISIBLE) < 1:
                            break
                        key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if cv2.getWindowProperty("Integrated inspection - Q or Escape to stop", cv2.WND_PROP_VISIBLE) < 1:
                    break
            if is_photo or args.max_frames is not None and processed >= args.max_frames:
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        for _, reader in sources:
            reader.close()
        if writer is not None:
            writer.release()
        if records is not None:
            records.close()
        if args.show:
            cv2.destroyAllWindows()
        run.update(processed_frames=processed, interrupted=interrupted)
        summary_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"Saved result: {media_path.resolve()}")
    print(f"Saved detections: {records_path.resolve()}")
    print(f"Saved model/domain summary: {summary_path.resolve()}")
    return 130 if interrupted else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
