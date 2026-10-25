import argparse
from ultralytics import YOLO

p = argparse.ArgumentParser(description='Run the custom aluminium-defect model on an image or video.')
p.add_argument('--model', required=True, help='Path to your trained best.pt or exported model folder')
p.add_argument('--source', required=True, help='Image, video, or camera index such as 0')
p.add_argument('--conf', type=float, default=0.35)
p.add_argument('--imgsz', type=int, default=640)
a = p.parse_args()
source = int(a.source) if a.source.isdigit() else a.source
model = YOLO(a.model)
model.predict(source=source, conf=a.conf, imgsz=a.imgsz, show=True, save=True)
