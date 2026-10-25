import unittest,types,sys
from unittest.mock import patch
import numpy as np
from run_inspection import SourceReader
class FakePicamera:
    def __init__(self,index):self.index=index;self.stopped=False;self.closed=False
    def create_video_configuration(self,**kwargs):self.config=kwargs;return kwargs
    def configure(self,config):pass
    def start(self):pass
    def capture_array(self,stream):return np.full((8,10,3),[10,20,30],np.uint8)
    def stop(self):self.stopped=True
    def close(self):self.closed=True
class PiSourceTests(unittest.TestCase):
    def test_csi_setup_color_order_and_cleanup(self):
        with patch.dict(sys.modules,{'picamera2':types.SimpleNamespace(Picamera2=FakePicamera)}):
            r=SourceReader('picamera2:0').open();ok,frame=r.read();camera=r.capture
            self.assertTrue(ok);self.assertEqual(frame[0,0].tolist(),[10,20,30]);self.assertEqual(camera.config['main']['format'],'RGB888');r.close();self.assertTrue(camera.stopped and camera.closed)
    def test_bad_index(self):
        with patch.dict(sys.modules,{'picamera2':types.SimpleNamespace(Picamera2=FakePicamera)}):
            with self.assertRaises(ValueError):SourceReader('picamera2:bad').open()
if __name__=='__main__':unittest.main()
