import argparse
import cv2
from ultralytics import YOLO

p = argparse.ArgumentParser(description='Run trained NCNN model against a USB camera on Raspberry Pi.')
p.add_argument('--model', default='./aluminium_defects_ncnn_model')
p.add_argument('--camera', type=int, default=0)
p.add_argument('--conf', type=float, default=0.35)
p.add_argument('--imgsz', type=int, default=640)
a = p.parse_args()
model = YOLO(a.model)
cap = cv2.VideoCapture(a.camera)
if not cap.isOpened():
    raise SystemExit(f'Cannot open camera index {a.camera}. Camera interface may need configuration.')
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print('Camera frame read failed; stopping.')
            break
        result = model.predict(frame, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
        view = result.plot()
        cv2.imshow('Aluminium Defect Detection — press q to quit', view)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
