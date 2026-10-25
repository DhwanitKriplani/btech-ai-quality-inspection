import unittest
from unittest.mock import Mock
import numpy as np
import torch
from background_localizer import BackgroundLocalizer
import run_inspection as app

class Classifier:
    def predict_many(self, images):
        return [{'prediction':'metal','score':.8,'scores':{'metal':.8,'plastic':.1,'wood':.1}} for _ in images]

class BackgroundTests(unittest.TestCase):
    def setUp(self):self.ref=np.full((120,160,3),80,np.uint8)
    def test_empty_has_no_boxes(self):
        self.assertEqual(len(BackgroundLocalizer(self.ref).predict(self.ref)[0].boxes.xyxy),0)
    def test_observed_object_geometry(self):
        frame=self.ref.copy();frame[30:80,40:110]=240
        r=BackgroundLocalizer(self.ref).predict(frame)[0]
        self.assertEqual(len(r.boxes.xyxy),1)
        self.assertTrue(np.all(np.abs(r.boxes.xyxy.numpy()[0]-[40,30,110,80])<=3))
    def test_noise_filtered_and_dimensions_checked(self):
        frame=self.ref.copy();frame[50,50]=255
        self.assertEqual(len(BackgroundLocalizer(self.ref).predict(frame)[0].boxes.xyxy),0)
        with self.assertRaises(ValueError):BackgroundLocalizer(self.ref).predict(frame[:100])
    def test_input_validation(self):
        with self.assertRaises(ValueError):BackgroundLocalizer(self.ref.astype(np.uint16))
        with self.assertRaises(ValueError):BackgroundLocalizer(self.ref,threshold=float('nan'))
    def pipeline(self):
        p=app.InspectionPipeline.__new__(app.InspectionPipeline)
        p.device='cpu';p.imgsz=p.material_imgsz=p.defect_imgsz=640;p.material_conf=.25;p.defect_conf=.35;p.max_materials=12;p.include_markers=False
        p.material_detectors={'rgb':BackgroundLocalizer(self.ref)};p.classifiers={'rgb':Classifier()}
        result=Mock();result.boxes=Mock(xyxy=torch.tensor([[60.,40.,90.,55.]]),conf=torch.tensor([.9]),cls=torch.tensor([3.]));result.masks=None;result.names={3:'scratch'}
        p.defect_detector=Mock();p.defect_detector.predict.return_value=[result]
        return p
    def test_grounded_objects_classification_and_defect_association(self):
        frame=self.ref.copy();frame[30:80,40:110]=240
        _,r=self.pipeline().process_frame(frame)
        self.assertEqual(len(r['materials']),1)
        m=r['materials'][0];self.assertEqual(m['material_estimate'],'metal');self.assertIsNone(m['detector_score']);self.assertIsNone(m['detector_label']);self.assertIsNone(m['classifier_agrees'])
        self.assertEqual(r['defects']['items'][0]['material_region_id'],'M1')
    def test_empty_scene_has_no_invented_object_or_background_defect(self):
        _,r=self.pipeline().process_frame(self.ref)
        self.assertEqual(r['materials'],[]);self.assertEqual(r['defects']['items'],[])
if __name__=='__main__':unittest.main()
