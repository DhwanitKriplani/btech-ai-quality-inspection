import argparse
from ultralytics import YOLO
p = argparse.ArgumentParser(description='Export trained custom model to NCNN for Raspberry Pi 5.')
p.add_argument('--model', required=True, help='Path to trained best.pt')
a = p.parse_args()
model = YOLO(a.model)
path = model.export(format='ncnn', imgsz=640)
print(f'NCNN export created: {path}')
