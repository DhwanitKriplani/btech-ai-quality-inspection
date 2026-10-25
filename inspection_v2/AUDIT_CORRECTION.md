# Correction to the original defect benchmark

The earlier approximately 99% AP report used the supplied 50-image test split. A read-only audit later confirmed that 8 images were byte-identical to training images and 2 were identical to validation images. That result is not fully independent unseen-image performance.

Across all 480 source images, 40 exact-RGB duplicate families had conflicting scratch polygons. All 80 copies in those families were conservatively excluded. The remaining 400 images were split into 282 training, 60 validation and 58 test images, with 27/8/10 disjoint capture groups. All four classes are present. Training restarted from generic 80-class COCO YOLO26n segmentation weights, not the old custom model.

No exact image hashes, source families or capture groups cross the new splits. Physical part IDs are unavailable, so capture groups are a proxy, not proof of unseen physical parts.

V2 measured 97.28% box mAP50 and 93.86% mask mAP50 on the new 58-image grouped test. These results apply to RGB aluminium surfaces, not the user's CCTV hardware. Original files and model weights are preserved; V2 uses the new weights. Full manifests and results are in evaluation/.
