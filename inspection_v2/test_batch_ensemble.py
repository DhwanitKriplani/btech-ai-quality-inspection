"""CPU-only synthetic/mocked checks for batch order and probability-mixture heads."""
import copy
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch
from torchvision import models

from material_predictor import MaterialPredictor, resolve_device
import run_inspection as app


class FeatureCapture(torch.nn.Module):
    def forward(self, batch):
        self.last_batch = batch.clone()
        return torch.zeros((len(batch), 512, 1, 1), dtype=batch.dtype, device=batch.device)


class TensorList:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class BatchEnsembleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(123)
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(512, 3)
        cls.state = model.state_dict()
        cls.images = [Image.new("RGB", (80, 100), color) for color in ((240, 20, 10), (10, 30, 230), (10, 210, 40))]
        cls.rgb_pre = {"resize": 256, "center_crop": 224, "color_mode": "rgb",
                       "mean": [.485, .456, .406], "std": [.229, .224, .225]}
        cls.gray_pre = dict(cls.rgb_pre, color_mode="grayscale", grayscale_channels=3)

    def checkpoint(self, ensemble=False):
        checkpoint = {"architecture": "resnet18", "class_names": ["metal", "plastic", "wood"],
                      "model_state_dict": self.state, "preprocessing": self.rgb_pre,
                      "training_domain": "mcubes_rgb_material_crops"}
        if ensemble:
            checkpoint["training_domain"] = "mcubes_rgb_and_fmd_material_crops"
            checkpoint["ensemble"] = {
                "type": "rgb_grayscale_shared_backbone", "rgb_weight": .5, "gray_weight": .5,
                "grayscale_fc_state_dict": {"weight": self.state["fc.weight"].clone() * -.3,
                                             "bias": torch.tensor([.2, -.1, .3])},
                "grayscale_preprocessing": self.gray_pre,
                "probability_combination": "weighted_softmax_average", "shared_backbone_exact_equal": True,
                "backbone_sha256": "synthetic-test-no-model-file"}
        return checkpoint

    def predictor(self, checkpoint):
        with patch.object(Path, "is_file", return_value=True), patch("material_predictor.torch.load", return_value=checkpoint):
            return MaterialPredictor(Path(__file__).with_name("mock_state_not_saved.pt"), threads=2)

    def test_plain_batch_matches_scalars_and_input_order(self):
        predictor = self.predictor(self.checkpoint())
        batch = predictor.predict_many(self.images)
        scalar = [predictor.predict(image) for image in self.images]
        for a, b in zip(batch, scalar):
            np.testing.assert_allclose(list(a["scores"].values()), list(b["scores"].values()), atol=2e-6, rtol=1e-5)
            self.assertEqual(a["prediction"], b["prediction"])
        self.assertFalse(np.allclose(list(batch[0]["scores"].values()), list(batch[1]["scores"].values())))
        self.assertEqual(predictor.predict_many([]), [])

    def test_two_input_color_modes_and_probability_average_not_logit_average(self):
        predictor = self.predictor(self.checkpoint(True))
        predictor.feature_backbone = FeatureCapture()
        with torch.no_grad():
            predictor.model.fc.weight.zero_()
            predictor.model.fc.bias.copy_(torch.tensor([3., 0., 0.]))
            predictor.gray_head.weight.zero_()
            predictor.gray_head.bias.copy_(torch.tensor([0., 0., 2.]))
        output = predictor.predict_many(self.images[:2])
        captured = predictor.feature_backbone.last_batch
        self.assertEqual(tuple(captured.shape), (4, 3, 224, 224))
        for index, image in enumerate(self.images[:2]):
            self.assertTrue(torch.equal(captured[index], predictor.preprocess(image)))
            self.assertTrue(torch.equal(captured[index + 2], predictor.gray_preprocess(image.convert("L"))))
        expected = .5 * torch.tensor([3., 0., 0.]).softmax(0) + .5 * torch.tensor([0., 0., 2.]).softmax(0)
        wrong = torch.tensor([1.5, 0., 1.]).softmax(0)
        self.assertFalse(torch.allclose(expected, wrong))
        for result in output:
            np.testing.assert_allclose(list(result["scores"].values()), expected.numpy(), atol=1e-7)
            self.assertEqual(result["score_type"], "uncalibrated_probability_mixture")
            self.assertIn("ensemble_components", result)

    def test_real_shared_backbone_batch_matches_scalar(self):
        predictor = self.predictor(self.checkpoint(True))
        before = {name: tensor.clone() for name, tensor in predictor.model.state_dict().items() if "num_batches_tracked" in name}
        batch = predictor.predict_many(self.images[:2])
        scalar = [predictor.predict(image) for image in self.images[:2]]
        for a, b in zip(batch, scalar):
            np.testing.assert_allclose(list(a["scores"].values()), list(b["scores"].values()), atol=2e-6, rtol=1e-5)
            self.assertEqual(a["prediction"], b["prediction"])
        for name, value in before.items():
            self.assertTrue(torch.equal(value, predictor.model.state_dict()[name]))

    def test_bad_ensemble_weights_and_head_shapes_rejected(self):
        cases = []
        for weights in ((-.1, 1.1), (float("nan"), .5), (.3, .3)):
            checkpoint = self.checkpoint(True)
            checkpoint["ensemble"].update(rgb_weight=weights[0], gray_weight=weights[1])
            cases.append(checkpoint)
        checkpoint = self.checkpoint(True)
        checkpoint["ensemble"]["grayscale_fc_state_dict"]["weight"] = torch.zeros((2, 512))
        cases.append(checkpoint)
        for checkpoint in cases:
            with self.assertRaises(ValueError):
                self.predictor(checkpoint)

    def test_device_parser_without_gpu_work(self):
        self.assertEqual(str(resolve_device("cpu")), "cpu")
        with patch("material_predictor.torch.cuda.is_available", return_value=False):
            self.assertEqual(str(resolve_device("auto")), "cpu")
            with self.assertRaises(RuntimeError):
                resolve_device("0")
        with patch("material_predictor.torch.cuda.is_available", return_value=True), patch("material_predictor.torch.cuda.device_count", return_value=2):
            for value in ("0", "cuda0", "cuda:0", "auto"):
                self.assertEqual(str(resolve_device(value)), "cuda:0")

    def test_pipeline_batches_only_actual_roi_crops_in_order(self):
        class BatchClassifier:
            def predict_many(self, images):
                self.colors = [image.getpixel((0, 0)) for image in images]
                self.calls = getattr(self, "calls", 0) + 1
                return [{"prediction": "metal", "score": .8, "scores": {"metal": .8, "plastic": .1, "wood": .1}} for _ in images]

            def predict(self, image):
                raise AssertionError("Actual ROI batch must not call scalar prediction")

        pipeline = app.InspectionPipeline.__new__(app.InspectionPipeline)
        pipeline.device, pipeline.imgsz = "cpu", 640
        pipeline.material_conf, pipeline.defect_conf, pipeline.max_materials = .25, .35, 12
        pipeline.include_markers = False
        result = Mock(names={0: "metal", 1: "plastic", 2: "wood"}, masks=None)
        result.boxes.xyxy = TensorList([[0, 0, 16, 32], [16, 0, 32, 32], [40, 0, 45, 10]])
        result.boxes.cls = TensorList([0, 0, 0])
        result.boxes.conf = TensorList([.8, .7, .6])
        pipeline.material_detectors = {"rgb": Mock()}
        pipeline.material_detectors["rgb"].predict.return_value = [result]
        classifier = BatchClassifier()
        pipeline.classifiers = {"rgb": classifier}
        pipeline.defect_detector = Mock()
        pipeline.defect_detector.predict.return_value = [Mock(boxes=None, masks=None)]
        frame = np.zeros((32, 32, 3), np.uint8)
        frame[:, :16] = [0, 0, 255]
        frame[:, 16:] = [255, 0, 0]
        _, report = pipeline.process_frame(frame)
        self.assertEqual(classifier.calls, 1)
        self.assertEqual(classifier.colors, [(255, 0, 0), (0, 0, 255)])
        self.assertEqual(len(report["materials"]), 2)
        self.assertEqual(report["materials"][1]["bbox_xyxy"], [16., 0., 32., 32.])


if __name__ == "__main__":
    unittest.main()
