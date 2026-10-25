"""Synthetic planar calibration checks; never opens a camera or GUI."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import calibrate_views as calibration
import run_inspection as app

HERE = Path(__file__).resolve().parent


def known_calibration():
    rgb = np.array([[10, 10], [70, 10], [70, 70], [10, 70], [35, 25], [25, 55]], float)
    return calibration.estimate_calibration(rgb, rgb + [10, 20], [100, 100], [120, 130])


def panel_fixture():
    rgb = {"original_size_wh": [100, 100], "materials": [],
           "defects": {"items": [{"detection_id": "D1", "label": "scratch",
                                    "bbox_xyxy": [20, 10, 40, 30], "material_region_id": "M_RGB",
                                    "mask_polygon_xy": [[20, 10], [40, 10], [40, 30], [20, 30]]}],
                       "surface_defect_count": 1, "marker_count": 0}}
    nir = {"original_size_wh": [120, 130],
           "materials": [{"region_id": "N_wide", "bbox_xyxy": [0, 0, 115, 125]},
                         {"region_id": "N_small", "bbox_xyxy": [35, 35, 50, 50]}],
           "defects": {"items": [], "surface_defect_count": 0, "marker_count": 0}}
    return [("rgb", np.zeros((142, 100, 3), np.uint8), rgb),
            ("nir", np.zeros((172, 120, 3), np.uint8), nir)]


class CalibrationTests(unittest.TestCase):
    def test_material_and_defect_detectors_receive_different_sizes(self):
        pipeline = app.InspectionPipeline.__new__(app.InspectionPipeline)
        pipeline.device, pipeline.imgsz = "cpu", 640
        pipeline.material_imgsz, pipeline.defect_imgsz = 1024, 640
        pipeline.material_conf, pipeline.defect_conf, pipeline.max_materials = .25, .35, 12
        pipeline.include_markers = False
        empty = Mock(boxes=None, masks=None)
        pipeline.material_detectors = {"rgb": Mock()}
        pipeline.material_detectors["rgb"].predict.return_value = [empty]
        pipeline.classifiers = {"rgb": Mock()}
        pipeline.defect_detector = Mock()
        pipeline.defect_detector.predict.return_value = [empty]
        pipeline.process_frame(np.zeros((48, 64, 3), np.uint8), "rgb")
        self.assertEqual(pipeline.material_detectors["rgb"].predict.call_args.kwargs["imgsz"], 1024)
        self.assertEqual(pipeline.defect_detector.predict.call_args.kwargs["imgsz"], 640)

    def test_known_projective_homography_and_xy_pixel_order(self):
        h = np.array([[1.1, .04, 7], [.01, .9, 9], [.0003, .0001, 1]])
        rgb = np.array([[10, 10], [150, 10], [150, 130], [10, 130], [70, 50], [80, 90]], float)
        nir = calibration.map_points(rgb, h)
        result = calibration.estimate_calibration(rgb, nir, [200, 160], [240, 180])
        np.testing.assert_allclose(calibration.map_points([[43, 62]], result["matrix"]),
                                   calibration.map_points([[43, 62]], h), atol=1e-4)
        np.testing.assert_allclose(calibration.map_points([[4, 9]], [[1, 0, 15], [0, 1, 7], [0, 0, 1]]), [[19, 16]])

    def test_collinear_bounds_and_nonfinite_points_rejected(self):
        for points in ([[0, 0], [10, 10], [20, 20], [30, 30]],
                       [[0, 0], [120, 0], [50, 40], [0, 40]],
                       [[0, 0], [20, 0], [20, 20], [float("nan"), 20]]):
            with self.assertRaises(ValueError):
                calibration.estimate_calibration(points, points, [100, 100], [100, 100])

    def test_singular_ill_conditioned_and_horizon_rejected(self):
        for h in ([[1, 0, 0], [0, 0, 0], [0, 0, 1]],
                  [[1, 0, 0], [0, 1e-12, 0], [0, 0, 1]],
                  [[1, 0, float("inf")], [0, 1, 0], [0, 0, 1]]):
            with self.assertRaises(ValueError):
                calibration.validate_matrix(h)
        with self.assertRaisesRegex(ValueError, "horizon"):
            calibration.map_points([[1, 2]], [[1, 0, 0], [0, 1, 0], [-1, 0, 1]])

    def test_dimensions_never_automatically_scaled(self):
        c = known_calibration()
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            calibration.verify_dimensions(c, [200, 200], [120, 130])
        panels = panel_fixture()
        panels[0][2]["original_size_wh"] = [101, 100]
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            app.calibrated_associations(panels, c)

    def test_association_preserves_rgb_and_maps_quadrilateral_mask(self):
        panels = panel_fixture()
        result = app.calibrated_associations(panels, known_calibration())
        association = result[0]
        np.testing.assert_allclose(association["mapped_center_xy"], [40, 40], atol=1e-5)
        np.testing.assert_allclose(association["mapped_box_polygon_xy"][0], [30, 30], atol=1e-5)
        self.assertEqual(association["nir_material_region_id"], "N_small")
        self.assertEqual(association["rgb_material_region_id"], "M_RGB")
        self.assertEqual(panels[0][2]["defects"]["items"][0]["material_region_id"], "M_RGB")
        self.assertEqual(association["score_fusion"], "none")
        self.assertEqual(len(association["mapped_mask_polygon_xy"]), 4)
        self.assertGreater(np.count_nonzero(panels[1][1]), 0)

    def test_tampered_matrix_and_direction_rejected_on_load(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory) / "calibration.json"
            c = known_calibration()
            for change in (lambda d: d.update(direction="nir_to_rgb"),
                           lambda d: d["matrix"][0].__setitem__(2, 45)):
                data = copy.deepcopy(c)
                change(data)
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    calibration.load_calibration(path)

    def test_points_cli_is_headless_and_read_photo_keeps_bgr_order(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            rgb, nir, points, output = [root / x for x in ("rgb.png", "nir.png", "points.json", "calibration.json")]
            Image.new("RGB", (100, 100), (255, 0, 0)).save(rgb)
            Image.new("L", (120, 130), 50).save(nir)
            c = known_calibration()
            points.write_text(json.dumps({"rgb": c["rgb_points_xy"], "nir": c["nir_points_xy"]}))
            with patch.object(calibration.cv2, "namedWindow", side_effect=AssertionError("GUI must not open")):
                calibration.main(["--rgb", str(rgb), "--nir", str(nir), "--points", str(points), "--output", str(output)])
            loaded = calibration.load_calibration(output)
            self.assertEqual(loaded["direction"], "rgb_to_nir")
            self.assertEqual(app.read_photo(rgb)[0, 0].tolist(), [0, 0, 255])

    def test_mock_pipeline_enables_only_explicit_calibration(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            rgb, nir, saved = [root / x for x in ("rgb.png", "nir.png", "calibration.json")]
            Image.new("RGB", (100, 100), "red").save(rgb)
            Image.new("L", (120, 130), 50).save(nir)
            saved.write_text(json.dumps(known_calibration()))
            def fake_pipeline():
                pipeline = Mock()
                pipeline.metadata.return_value = {"models": {}, "limitations": []}
                pipeline.process_frame.side_effect = lambda frame, mode: next((image, report) for name, image, report in panel_fixture() if name == mode)
                return pipeline
            for use_calibration in (False, True):
                output = root / ("calibrated" if use_calibration else "uncalibrated")
                argv = ["--source", str(rgb), "--nir-source", str(nir), "--output", str(output)]
                if use_calibration:
                    argv += ["--calibration", str(saved)]
                with patch.object(app, "InspectionPipeline", return_value=fake_pipeline()), patch.object(app.cv2, "imshow", side_effect=AssertionError("No GUI")):
                    self.assertEqual(app.main(argv), 0)
                report = json.loads(next(p for p in output.glob("*.json") if not p.name.endswith(".summary.json")).read_text())
                self.assertEqual(report["sensor_registration"], "planar_homography_rgb_to_nir" if use_calibration else "not_registered")
                if use_calibration:
                    self.assertEqual(report["frame"]["cross_sensor_associations"][0]["nir_material_region_id"], "N_small")
                    self.assertIn("reprojection", report["calibration"])
                    self.assertEqual(report["capture_synchronization"], "sequential_reads_not_hardware_synchronized")
                else:
                    self.assertIsNone(report["frame"]["cross_sensor_associations"])

    def test_cli_stage_defaults_and_explicit_common_override(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            photo = root / "rgb.png"
            Image.new("RGB", (100, 100), "red").save(photo)
            for override, expected in (([], (640, 640)), (["--imgsz", "512"], (512, 512))):
                pipeline = Mock()
                pipeline.metadata.return_value = {"models": {}}
                image, report = panel_fixture()[0][1:]
                pipeline.process_frame.return_value = (image, report)
                with patch.object(app, "InspectionPipeline", return_value=pipeline) as constructor:
                    app.main(["--source", str(photo), "--output", str(root / str(expected[0]))] + override)
                self.assertEqual(constructor.call_args.kwargs["material_imgsz"], expected[0])
                self.assertEqual(constructor.call_args.kwargs["defect_imgsz"], expected[1])


if __name__ == "__main__":
    unittest.main()
