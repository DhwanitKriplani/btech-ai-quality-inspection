"""Check RGB preservation and legacy grayscale preprocessing using random mocked state."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms

from material_predictor import MaterialPredictor


class CaptureInput(torch.nn.Module):
    def forward(self, batch):
        self.last_input = batch.clone()
        return torch.zeros((len(batch), 3))


class ColorModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(512, 3)
        cls.state = model.state_dict()
        values = np.zeros((80, 100, 3), np.uint8)
        values[:, :50, 0] = 255
        values[:, 50:, 2] = 220
        values[:, :, 1] = np.arange(100, dtype=np.uint8)[None, :]
        cls.image = Image.fromarray(values)

    def predictor(self, metadata, domain):
        checkpoint = {"architecture": "resnet18", "class_names": ["metal", "plastic", "wood"],
                      "training_domain": domain, "preprocessing": metadata, "model_state_dict": self.state}
        with patch.object(Path, "is_file", return_value=True), patch("material_predictor.torch.load", return_value=checkpoint):
            predictor = MaterialPredictor(Path(__file__).with_name("mock_state_not_saved.pt"), threads=2)
        predictor.model = CaptureInput()
        return predictor

    def test_rgb_keeps_color_without_grayscale_channels_metadata(self):
        predictor = self.predictor({"resize": 256, "center_crop": 224, "color_mode": "rgb",
                                    "mean": [.485, .456, .406], "std": [.229, .224, .225]},
                                   "mcubes_rgb_material_crops")
        result = predictor.predict(self.image)
        expected = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                       transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])(self.image)
        self.assertTrue(torch.equal(predictor.model.last_input[0], expected))
        raw = expected * torch.tensor([.229, .224, .225])[:, None, None] + torch.tensor([.485, .456, .406])[:, None, None]
        self.assertFalse(torch.allclose(raw[0], raw[2]))
        self.assertIn("MCubeS RGB", result["scope"])

    def test_legacy_default_retains_exact_grayscale_before_resize_order(self):
        predictor = self.predictor({"resize": 256, "center_crop": 224, "grayscale_channels": 3}, "grayscale_fmd")
        predictor.predict(self.image)
        expected = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                       transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                       transforms.Normalize([.485, .456, .406], [.229, .224, .225])])(self.image.convert("L"))
        self.assertTrue(torch.equal(predictor.model.last_input[0], expected))
        self.assertEqual(predictor.color_mode, "grayscale")


if __name__ == "__main__":
    unittest.main()
