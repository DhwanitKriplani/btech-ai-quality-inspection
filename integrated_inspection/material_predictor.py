"""Predict metal/plastic/wood from an image or an explicitly selected camera."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageOps
import torch
from torchvision import models, transforms


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / ("best_nir.pt" if (HERE / "best_nir.pt").is_file() else "best_material.pt")
FMD_SCOPE = "Grayscale FMD classifier; scores are uncalibrated and do not verify NIR performance."


class MaterialPredictor:
    """Reusable CPU predictor. ``predict`` accepts a PIL image or an image path."""

    def __init__(self, model_path=DEFAULT_MODEL, threads=min(4, os.cpu_count() or 1)):
        if not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer")
        torch.set_num_threads(threads)
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("Expected the checkpoint dictionary saved by train_material.py")
        if checkpoint.get("architecture", "resnet18") != "resnet18":
            raise ValueError("This predictor supports ResNet18 checkpoints only")
        self.training_domain = checkpoint.get("training_domain", "grayscale_fmd")
        self.dataset_name = checkpoint.get("dataset_name", self.training_domain)
        self.dataset_source = checkpoint.get("dataset_source", "")
        self.sensor_name = checkpoint.get("sensor_name")
        self.source_verification = checkpoint.get("source_verification")
        if self.training_domain == "mcubes_nir_material_crops":
            self.scope = ("Trained on MCubeS NIR material crops; scores are uncalibrated. "
                          "CCTV performance requires separate testing.")
            self.scope_short = "MCubeS NIR crops; scores uncalibrated; CCTV untested"
        elif self.training_domain == "user_supplied_nir_camera_images":
            self.scope = ("Trained on user-supplied NIR camera images; scores uncalibrated; "
                          "see held-out metrics. Transfer to other setups is not guaranteed.")
            self.scope_short = "User-supplied NIR training; scores uncalibrated"
        elif self.training_domain == "grayscale_fmd":
            self.scope = FMD_SCOPE
            self.scope_short = "Grayscale FMD; scores uncalibrated; NIR unverified"
        else:
            self.scope = (f"Checkpoint training domain: {self.training_domain}; scores are "
                          "uncalibrated. See held-out metrics for the applicable camera setup.")
            self.scope_short = "Checkpoint-specific training; scores uncalibrated"
        self.class_names = checkpoint.get("class_names", ["metal", "plastic", "wood"])
        if (not isinstance(self.class_names, (list, tuple)) or len(self.class_names) < 2
                or not all(isinstance(name, str) and name for name in self.class_names)
                or len(set(self.class_names)) != len(self.class_names)):
            raise ValueError("Checkpoint class_names must contain unique nonempty strings")
        metadata = checkpoint.get("preprocessing", {})
        if not isinstance(metadata, dict):
            raise ValueError("Invalid checkpoint preprocessing metadata")
        self.resize = metadata.get("resize", 256)
        self.center_crop = metadata.get("center_crop", 224)
        if (not isinstance(self.resize, int) or not isinstance(self.center_crop, int)
                or self.resize < 1 or not 0 < self.center_crop <= self.resize):
            raise ValueError("Expected positive integer resize and center_crop <= resize")
        if metadata.get("grayscale_channels", 3) != 3:
            raise ValueError("ResNet18 requires grayscale repeated into three channels")
        mean = metadata.get("mean", [0.485, 0.456, 0.406])
        std = metadata.get("std", [0.229, 0.224, 0.225])
        if len(mean) != 3 or len(std) != 3 or any(float(v) <= 0 for v in std):
            raise ValueError("Expected three normalization means and positive standard deviations")
        self.preprocess = transforms.Compose([
            transforms.Resize(self.resize), transforms.CenterCrop(self.center_crop),
            transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        # weights=None ensures inference never downloads ImageNet weights.
        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()

    @staticmethod
    def _check_image(image, path=None):
        """Preserve sensor intensity meaning by refusing implicit 16-bit clipping."""
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Use a single camera frame per image file")
        high_depth = image.mode in {"I", "F"} or image.mode.startswith("I;")
        if image.format == "PNG":
            header = b""
            filename = path or getattr(image, "filename", None)
            if filename:
                with Path(filename).open("rb") as stream:
                    header = stream.read(25)
            else:
                stream = getattr(image, "fp", None)
                if stream is not None and stream.seekable():
                    position = stream.tell()
                    try:
                        stream.seek(0)
                        header = stream.read(25)
                    finally:
                        stream.seek(position)
            high_depth = high_depth or (len(header) == 25 and header[24] > 8)
        if hasattr(image, "tag_v2"):
            bits = image.tag_v2.get(258, (8,))
            if isinstance(bits, int):
                bits = (bits,)
            high_depth = high_depth or any(bit > 8 for bit in bits)
        if high_depth:
            raise ValueError("16-bit/float/raw intensities require fixed sensor normalization; "
                             "export consistently normalized 8-bit camera images first")
        if image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "YCbCr"}:
            raise ValueError(f"Unsupported camera image mode: {image.mode}")

    @staticmethod
    def read_image(source):
        if isinstance(source, Image.Image):
            MaterialPredictor._check_image(source)
            return ImageOps.exif_transpose(source).convert("RGB")
        path = Path(source)
        with Image.open(path) as image:
            MaterialPredictor._check_image(image, path)
            return ImageOps.exif_transpose(image).convert("RGB")

    def predict(self, source):
        image = self.read_image(source)
        # Both training loaders convert to L before resizing; preserve that order.
        batch = self.preprocess(image.convert("L")).unsqueeze(0)
        with torch.inference_mode():
            scores = self.model(batch).softmax(dim=1)[0].tolist()
        index = max(range(len(scores)), key=scores.__getitem__)
        return {
            "prediction": self.class_names[index], "score": scores[index],
            "scores": dict(zip(self.class_names, scores)),
            "score_type": "uncalibrated_softmax", "scope": self.scope,
            "training_domain": self.training_domain, "sensor_name": self.sensor_name,
            "dataset_name": self.dataset_name, "dataset_source": self.dataset_source,
            "source_verification": self.source_verification,
        }

    def input_region(self, width, height):
        """Central square in original frame coordinates, not an object box."""
        side = min(width, height) * self.center_crop / self.resize
        return (round((width - side) / 2), round((height - side) / 2),
                round((width + side) / 2), round((height + side) / 2))


def print_result(result):
    print(f"Prediction: {result['prediction']}")
    print("Class scores (uncalibrated softmax; not measured accuracy):")
    for name, score in result["scores"].items():
        print(f"  {name}: {score:.2%}")
    print(result["scope"])
    if result.get("sensor_name"):
        print(f"Training sensor: {result['sensor_name']}")
    if result.get("source_verification"):
        print(result["source_verification"])


def annotate_image(image, result):
    """Add a readable caption without implying any object localization."""
    image = image.convert("RGB")
    lines = [f"Prediction: {result['prediction']}",
             " | ".join(f"{name}: {score:.1%}" for name, score in result["scores"].items()),
             result["scope"]]
    probe = ImageDraw.Draw(image)
    width = max(image.width, max(probe.textbbox((0, 0), line)[2] for line in lines) + 20)
    canvas = Image.new("RGB", (width, image.height + 70), "#151b24")
    canvas.paste(image, ((width - image.width) // 2, 0))
    draw = ImageDraw.Draw(canvas)
    for offset, line in enumerate(lines):
        draw.text((10, image.height + 7 + 19 * offset), line, fill="white")
    return canvas


def save_results(args, predictor, image, result):
    protected = {predictor.model_path}
    if args.source:
        protected.add(Path(args.source).resolve())
    destinations = [Path(p).resolve() for p in (args.output, args.json) if p]
    if any(path in protected for path in destinations) or len(set(destinations)) != len(destinations):
        raise ValueError("Output paths must differ from the model, source image, and each other")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        annotate_image(image, result).save(output)
        print(f"Saved annotated image: {output.resolve()}")
    if args.json:
        report = dict(result, model=str(predictor.model_path),
                      source=str(args.source) if args.source else f"camera:{args.camera}")
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report: {output.resolve()}")


def run_camera(args, predictor):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Camera mode needs opencv-python: install it using your environment's python -m pip install opencv-python") from exc
    capture = cv2.VideoCapture(args.camera)
    last_image, last_result = None, None
    window_name = "Material classifier - q or Esc to quit"
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open camera {args.camera}; close other camera apps and check camera permissions")
        print("Camera started. Press q or Esc in the camera window to quit.")
        print("The square shows the classifier input region; it is not an object detection.")
        print(predictor.scope)
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the camera")
            last_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            last_result = predictor.predict(last_image)
            display = frame.copy()
            height, width = display.shape[:2]
            x1, y1, x2, y2 = predictor.input_region(width, height)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 255), 2)
            labels = [f"Prediction: {last_result['prediction']}",
                      " | ".join(f"{name}: {score:.1%}" for name, score in last_result["scores"].items()),
                      predictor.scope_short,
                      "Square = classifier input region (not localization)"]
            for row, label in enumerate(labels):
                y = 25 + row * 23
                cv2.putText(display, label, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(display, label, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    if last_result is not None:
        print_result(last_result)
        save_results(args, predictor, last_image, last_result)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def camera_index(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("camera index must be zero or greater")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=Path, help="Path to a single image")
    group.add_argument("--camera", type=camera_index, help="Explicit camera index, e.g. 0")
    parser.add_argument("--output", type=Path, help="Optional annotated image; camera mode saves the last frame")
    parser.add_argument("--json", type=Path, help="Optional JSON report; camera mode saves the last prediction")
    parser.add_argument("--threads", type=positive_int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    predictor = MaterialPredictor(args.model, threads=args.threads)
    if args.source:
        image = predictor.read_image(args.source)
        result = predictor.predict(image)
        print_result(result)
        save_results(args, predictor, image, result)
    else:
        run_camera(args, predictor)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
