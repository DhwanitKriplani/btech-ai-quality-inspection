# V2 measured results

| Component | Test | Result |
| --- | --- | --- |
| RGB ensemble classifier | FMD, 137 photos | 77.37% accuracy; macro F1 0.77128 |
| Same RGB classifier | MCubeS RGB, 455 crops | 74.29% accuracy; macro F1 0.64148 |
| Unchanged NIR classifier | MCubeS NIR, 455 crops | 68.35% accuracy; macro F1 0.51875 |
| Clean aluminium defect boxes | 58 capture-grouped RGB test images | mAP50 97.28%; mAP50–95 67.66% |
| Clean defect masks | Same 58 images | mAP50 93.86%; mAP50–95 46.36% |
| Scratch class alone | Same 58 images | Box AP50 97.23%; mask AP50 95.19% |
| Unchanged learned RGB material boxes | 102 MCubeS frames | mAP50 11.76%; mAP50–95 5.48% |
| Unchanged learned NIR material boxes | 102 MCubeS frames | mAP50 9.86%; mAP50–95 4.90% |

Classification accuracy, AP and pixel IoU are different metrics. No whole-system or Pi-camera accuracy is claimed. Material-region boxes remain weak, especially for plastic. Empty-background localization is a separate fixed-camera image-difference method; no confidence or mAP is fabricated for it.

The RGB ensemble was frozen using two validation domains before one-time evaluation of both held-out tests. It correctly classifies 106/137 FMD images, compared with 102/137 for the previous grayscale baseline. Tradeoffs remain by class. MCubeS RGB plastic recall is 16/57 (28.07%); NIR plastic recall remains 4/57 (7.02%). Converting RGB to grayscale does not create NIR data.

A larger YOLO11s/1024 candidate and a pixel-model NIR candidate failed the declared validation improvement criterion, so their weights were not deployed. Retained YOLO material and scratch models default to 640-pixel inference; classifier crops use 224. Classifiers run in ordered CPU/CUDA batches with raw estimates preserved.

The clean 400-image defect dataset excludes 80 ambiguous duplicate copies. The new 58-image grouped test was evaluated once after validation-based model selection. See AUDIT_CORRECTION.md for why the old contaminated result is superseded.

No Raspberry Pi hardware, real CCTV frames, physical calibration, FPS, thermal or 4 GB RAM benchmark has been tested. PC software checks and mathematical calibration checks are under review/.
