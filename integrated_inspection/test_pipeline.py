"""Focused integration checks; no cameras, downloads or model training."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import json

import cv2
import numpy as np
from PIL import Image

import run_inspection as app


class FakeTensor:
    def __init__(self, data):
        self.data = data

    def cpu(self):
        return self

    def tolist(self):
        return self.data


def prediction(boxes, labels, scores, names, polygons=None):
    result = Mock()
    result.names = dict(enumerate(names))
    result.boxes.xyxy = FakeTensor(boxes)
    result.boxes.cls = FakeTensor(labels)
    result.boxes.conf = FakeTensor(scores)
    result.masks = None if polygons is None else Mock(xy=polygons)
    return result


def pipeline(modality="rgb", include_markers=False):
    p = app.InspectionPipeline.__new__(app.InspectionPipeline)
    p.device, p.imgsz, p.material_conf, p.defect_conf = "cpu", 640, .25, .35
    p.max_materials, p.include_markers = 12, include_markers
    p.material_detectors = {modality: Mock()}
    p.classifiers = {modality: Mock()}
    p.defect_detector = Mock()
    p.classifiers[modality].predict.return_value = {
        "prediction": "plastic", "score": .6,
        "scores": {"metal": .3, "plastic": .6, "wood": .1}}
    return p


class PipelineTests(unittest.TestCase):
    def test_real_boxes_and_disagreement_stay_separate(self):
        p = pipeline()
        p.material_detectors["rgb"].predict.return_value = [prediction(
            [[-10, 0, 130, 100], [30, 20, 70, 70]], [0, 2], [.8, .7], app.MATERIAL_NAMES)]
        polygon = np.array([[40, 30], [60, 30], [60, 40], [40, 40]], np.float32)
        p.defect_detector.predict.return_value = [prediction(
            [[40, 30, 60, 40], [110, 80, 119, 90]], [3, 1], [.9, .5], app.DEFECT_NAMES,
            [polygon, polygon])]
        output, report = p.process_frame(np.zeros((100, 120, 3), np.uint8))
        self.assertEqual(report["materials"][0]["bbox_xyxy"], [0, 0, 120, 100])
        self.assertEqual(report["materials"][0]["detector_label"], "metal")
        self.assertEqual(report["materials"][0]["classifier"]["prediction"], "plastic")
        self.assertFalse(report["materials"][0]["classifier_agrees"])
        self.assertEqual(report["defects"]["items"][0]["material_region_id"], "M2")
        self.assertEqual(report["defects"]["surface_defect_count"], 2)
        self.assertEqual(output.shape, (142, 120, 3))
        self.assertEqual(p.defect_detector.predict.call_args.kwargs["classes"], [1, 2, 3])

    def test_no_detections_does_not_create_placeholder_boxes(self):
        p = pipeline()
        p.material_detectors["rgb"].predict.return_value = [prediction([], [], [], app.MATERIAL_NAMES)]
        p.defect_detector.predict.return_value = [prediction([], [], [], app.DEFECT_NAMES)]
        _, report = p.process_frame(np.zeros((32, 40, 3), np.uint8))
        self.assertEqual(report["materials"], [])
        self.assertEqual(report["defects"]["items"], [])
        self.assertEqual(report["inspection_status"], "no_defect_detected")
        p.classifiers["rgb"].predict.assert_not_called()

    def test_nir_never_runs_rgb_defect_model(self):
        p = pipeline("nir")
        p.material_detectors["nir"].predict.return_value = [prediction([], [], [], app.MATERIAL_NAMES)]
        _, report = p.process_frame(np.zeros((32, 40), np.uint8), "nir")
        self.assertEqual(report["defects"]["status"], "not_run_wrong_modality")
        p.defect_detector.predict.assert_not_called()

    def test_markers_are_separate_and_unassigned_defects_remain_unassigned(self):
        p = pipeline(include_markers=True)
        p.material_detectors["rgb"].predict.return_value = [prediction([], [], [], app.MATERIAL_NAMES)]
        p.defect_detector.predict.return_value = [prediction(
            [[1, 2, 20, 22], [21, 3, 30, 15]], [0, 3], [.8, .9], app.DEFECT_NAMES)]
        _, report = p.process_frame(np.zeros((32, 40, 3), np.uint8))
        self.assertEqual(report["defects"]["surface_defect_count"], 1)
        self.assertEqual(report["defects"]["marker_count"], 1)
        self.assertTrue(all(item["material_region_id"] is None for item in report["defects"]["items"]))

    def test_dual_reports_keep_original_coordinates(self):
        rgb = {"materials": [{"bbox_xyxy": [5, 8, 15, 20]}]}
        nir = {"materials": [{"bbox_xyxy": [2, 3, 7, 9]}]}
        combined, reports = app.compose_panels([
            ("rgb", np.zeros((142, 120, 3), np.uint8), rgb),
            ("nir", np.zeros((84, 80, 3), np.uint8), nir)])
        self.assertEqual(reports["rgb"]["materials"][0]["bbox_xyxy"], [5, 8, 15, 20])
        self.assertEqual(reports["nir"]["materials"][0]["bbox_xyxy"], [2, 3, 7, 9])
        self.assertGreater(reports["nir"]["display_transform"]["offset_x"], 0)
        self.assertEqual(combined.shape[0], 142)

    def test_high_depth_photos_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sensor.png"
            Image.fromarray(np.zeros((10, 10), np.uint16)).save(path)
            with self.assertRaisesRegex(ValueError, "fixed sensor normalization"):
                app.read_photo(path)

    def test_bad_and_outside_boxes_are_not_fabricated(self):
        self.assertIsNone(app.clip_box([30, 2, 40, 10], 20, 20))
        self.assertIsNone(app.clip_box([0, 0, float("nan"), 5], 20, 20))
        self.assertIsNone(app.associate_region([2, 2, 4, 4], []))

    @staticmethod
    def file_pipeline():
        p = Mock()
        p.metadata.return_value = {"models": {}, "detector_device": "cpu"}

        def process(frame, modality):
            frame = app.safe_frame(frame)
            output = np.zeros((frame.shape[0] + app.HEADER_HEIGHT, frame.shape[1], 3), np.uint8)
            output[app.HEADER_HEIGHT:] = frame
            return output, {"modality": modality, "original_size_wh": [frame.shape[1], frame.shape[0]],
                            "materials": [], "defects": {"surface_defect_count": 0, "marker_count": 0}}

        p.process_frame.side_effect = process
        return p

    def test_paired_photo_outputs_explain_separate_sensor_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb, nir = root / "rgb.png", root / "nir.png"
            Image.new("RGB", (40, 30), "red").save(rgb)
            Image.new("L", (28, 20), 80).save(nir)
            with patch.object(app, "InspectionPipeline", return_value=self.file_pipeline()):
                code = app.main(["--source", str(rgb), "--nir-source", str(nir), "--output", str(root / "out")])
            self.assertEqual(code, 0)
            result_path = next(p for p in (root / "out").glob("*.json") if not p.name.endswith(".summary.json"))
            result = json.loads(result_path.read_text())
            self.assertEqual(result["sensor_registration"], "not_registered")
            self.assertIsNone(result["frame"]["cross_sensor_associations"])
            self.assertEqual(result["frame"]["panels"]["rgb"]["original_size_wh"], [40, 30])
            self.assertEqual(result["frame"]["panels"]["nir"]["original_size_wh"], [28, 20])
            self.assertTrue(next((root / "out").glob("*.jpg")).is_file())

    def test_video_stride_outputs_readable_video_and_matching_json_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 12, (64, 48))
            if not writer.isOpened():
                self.skipTest("Local OpenCV build has no MJPG video writer")
            for value in (20, 40, 60, 80, 100):
                writer.write(np.full((48, 64, 3), value, np.uint8))
            writer.release()
            with patch.object(app, "InspectionPipeline", return_value=self.file_pipeline()):
                code = app.main(["--source", str(video), "--stride", "2", "--max-frames", "2",
                                 "--output", str(root / "out")])
            self.assertEqual(code, 0)
            records = [json.loads(line) for line in next((root / "out").glob("*.jsonl")).read_text().splitlines()]
            self.assertEqual([row["source_frame_index"] for row in records], [0, 2])
            capture = cv2.VideoCapture(str(next((root / "out").glob("*.mp4"))))
            try:
                decoded = 0
                while capture.read()[0]:
                    decoded += 1
            finally:
                capture.release()
            self.assertEqual(decoded, 2)


if __name__ == "__main__":
    unittest.main()
