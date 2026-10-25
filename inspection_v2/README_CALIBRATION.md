# Optional RGB-to-NIR calibration

This is a staged copy for review. It does not change the installed project and includes no model weights or physical camera calibration.

Use photos from the **same fixed camera setup and working plane**. Identify at least four noncollinear corresponding points, preferably more spread across the inspection area. Coordinates are zero-based `[x,y]` pixels in the EXIF-corrected images; RGB points and NIR points must describe the same physical features in the same order.

Save pairs as JSON:

```json
{"rgb": [[10,10], [70,10], [70,70], [10,70]], "nir": [[20,30], [80,30], [80,90], [20,90]]}
```

These are **synthetic example coordinates**, not a calibration for the user's camera.

Headless estimation:

```powershell
python calibrate_views.py --rgb rgb_photo.png --nir nir_photo.png --points pairs.json --output calibration.json
```

Omit `--points` only when deliberately opening the interactive photo windows. Click matching locations in both images in the same order, then press Enter. Esc/window-close aborts without saving. No webcam is opened by the calibrator.

Use the result with paired inputs:

```powershell
python run_inspection.py --source rgb_photo.png --nir-source nir_photo.png --calibration calibration.json --model-dir PATH_TO_EXISTING_MODELS
```

The matrix direction is RGB→NIR. Both frame sizes must exactly match the saved calibration; inputs are never automatically scaled to fit a calibration. Nonfinite/singular/ill-conditioned matrices, bad correspondences, unstable horizon projections and excessive inlier reprojection error are rejected. Saved residuals are recomputed on load.

When calibrated, RGB defect centres, box quadrilaterals and available mask polygons are projected into NIR source coordinates. The smallest NIR material region containing a mapped centre is recorded as a geometric association. The original RGB material-region association remains intact. A mapped RGB outline is drawn in magenta on the NIR pane; it is not an independently detected NIR scratch. Detector/classifier scores are not fused.

Without `--calibration`, the existing unregistered side-by-side behavior remains, and cross-sensor associations stay null.

**Limits:** a planar homography is valid only for its calibrated plane/depth and unchanged camera positions, lens settings, resolution and image processing. Different depths can cause parallax errors. Reprojection error on fitting points does not independently prove physical accuracy. This feature does not establish camera synchronization; dual streams are still read sequentially. Real calibration and testing on the user's camera setup remain necessary.

`material_predictor.py` also supports checkpoint preprocessing `color_mode="rgb"` for true three-channel RGB crops. Missing `color_mode` retains the exact previous grayscale-before-resize behavior. No weights are changed by that compatibility update.

The staged v2 runtime uses `--material-imgsz 1024` and `--defect-imgsz 640` by default. This preserves the scratch model's tested size while using the larger material detector's training size. An explicit legacy `--imgsz 512`, for example, overrides **both** stages to512. Classifier crops retain their own checkpoint-defined preprocessing (normally224). Both actual detector sizes are recorded in output metadata.

Checks: `python test_calibration.py` and `python test_color_mode.py` use synthetic geometry/mocked model state only, with no GUI, camera access or fabricated trained checkpoint.
