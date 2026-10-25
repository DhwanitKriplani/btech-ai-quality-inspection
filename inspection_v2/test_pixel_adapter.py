"""No-weight/no-CUDA checks for native semantic-component adapter contracts."""
import copy
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch

import material_pixel_adapter as adapter
import run_inspection as app


class PixelAdapterTests(unittest.TestCase):
    @staticmethod
    def checkpoint(modality="rgb"):
        return {"architecture": "lraspp_mobilenet_v3_large", "class_names": ["metal", "plastic", "wood", "other"],
                "modality": modality, "component_rules": dict(adapter.COMPONENT_RULES),
                "preprocess": {"mean": adapter.MEAN, "std": adapter.STD, "color": "RGB", "resize": None}}

    @staticmethod
    def result():
        return {"boxes": np.array([[1, 2, 20, 30], [25, 4, 50, 34]], np.float32),
                "scores": np.array([.8, .6], np.float32), "classes": np.array([0, 2]),
                "component_areas": np.array([300, 350]), "pixel_mask": np.zeros((40, 60), np.uint8)}

    def make_adapter(self, checkpoint):
        with patch.object(adapter, "load_pixel_checkpoint", return_value=(object(), checkpoint)):
            return adapter.MaterialPixelAdapter("mock_not_saved.pt", device="cpu", expected_modality=checkpoint["modality"])

    def test_rgb_pixel_order_native_coordinates_rules_tensors_and_maxdet(self):
        localizer = self.make_adapter(self.checkpoint())
        frame = np.full((40, 60, 3), [10, 20, 200], np.uint8)
        with patch.object(adapter, "predict_material_regions", return_value=self.result()) as predict:
            result = localizer.predict(frame, conf=.25, imgsz=1024, device="cpu", max_det=1)[0]
        pixels = predict.call_args.args[1]
        self.assertEqual(pixels.shape, frame.shape)
        self.assertEqual(pixels[0, 0].tolist(), [200, 20, 10])
        self.assertEqual(predict.call_args.kwargs["rules"]["confidence"], .5)
        self.assertEqual(predict.call_args.kwargs["rules"]["max_boxes"], 1)
        self.assertEqual(result.boxes.xyxy.tolist(), [[1., 2., 20., 30.]])
        self.assertEqual(result.boxes.cls.tolist(), [0.])
        self.assertEqual(result.boxes.xyxy.device.type, "cpu")
        self.assertEqual(result.input_resolution, "native")

    def test_nir_repeats_intensity_and_cli_confidence_is_not_loosened(self):
        localizer = self.make_adapter(self.checkpoint("nir"))
        frame = np.full((40, 60, 3), [10, 20, 200], np.uint8)
        with patch.object(adapter, "predict_material_regions", return_value=self.result()) as predict:
            result = localizer.predict(frame, conf=.75, max_det=12)[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(predict.call_args.args[1], np.repeat(gray[:, :, None], 3, 2))
        self.assertEqual(predict.call_args.kwargs["rules"]["confidence"], .75)
        self.assertEqual(len(result.boxes.xyxy), 1)

    def test_bad_architecture_class_order_mode_and_size_fail(self):
        for key, value in (("architecture", "resnet18"), ("class_names", ["metal", "wood", "plastic", "other"]),
                           ("modality", "thermal")):
            checkpoint = self.checkpoint()
            checkpoint[key] = value
            with patch.object(adapter, "load_pixel_checkpoint", return_value=(object(), checkpoint)), self.assertRaises(ValueError):
                adapter.MaterialPixelAdapter("mock.pt", device="cpu", expected_modality="rgb")
        checkpoint = self.checkpoint()
        checkpoint["preprocess"]["resize"] = 512
        with self.assertRaises(ValueError):
            self.make_adapter(checkpoint)

    def test_no_component_does_not_create_material_boxes(self):
        localizer = self.make_adapter(self.checkpoint())
        result = self.result()
        result.update(boxes=np.empty((0, 4), np.float32), scores=np.empty(0, np.float32),
                      classes=np.empty(0, np.int64), component_areas=np.empty(0, np.int64))
        with patch.object(adapter, "predict_material_regions", return_value=result):
            boxes = localizer.predict(np.zeros((40, 60, 3), np.uint8))[0].boxes
        self.assertEqual(tuple(boxes.xyxy.shape), (0, 4))
        self.assertEqual(len(boxes.cls), 0)

    def test_explicit_mismatched_cached_device_fails_without_cuda_work(self):
        localizer = self.make_adapter(self.checkpoint())
        with patch.object(adapter, "resolve_device", return_value=torch.device("cuda:0")), self.assertRaisesRegex(ValueError, "cached model device"):
            localizer.predict(np.zeros((40, 60, 3), np.uint8), device="0")


if __name__ == "__main__":
    unittest.main()
