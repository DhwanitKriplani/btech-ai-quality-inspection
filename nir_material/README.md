# Trained NIR material classifier

**`best_nir.pt` is trained and included.** It recognizes metal, plastic and wood from image crops using a CPU ResNet18 classifier. Training used the actual NIR camera channel of the public [MCubeS dataset](https://github.com/kyotovision-public/multimodal-material-segmentation). The user's CCTV camera samples will be tested later.

Read `START_HERE.txt` for Hinglish instructions and `MODEL_REPORT.md` for measured results and limitations.

## Results and limits

The public NIR test result is **68.35% accuracy (311/455)** and macro F1 **0.5187**. Metal recall is 84.00%, wood recall 65.54%, and **plastic recall is weak: 7.02% (4/57)**. This is a first trained NIR model, not a reliable production sorter. These are crop-classification results, not the original MCubeSNet segmentation benchmark.

Training used 931 crops from 212 source frames; validation used 293 crops from 76 frames; test used 455 crops from all 102 official test frames. Training, validation and test scene prefixes are disjoint. Overlapping scenes were purged from the original training/validation lists without moving any test frames into training. The crop manifest records the full derivation. All 100 head-training epochs completed, with epoch 1 chosen by minimum validation loss; the selected checkpoint was saved before test predictions were calculated.

The source NIR camera is a FLIR GS3-U3-41C6NIR-C with a 750 nm long-pass filter. The user's description is “3.6mm IR 1080P CCTV module”; no images from it were supplied. **Accuracy on that separate camera remains unknown.** Only the Integrated Webcam was reported by Windows during preparation, and no physical camera was opened during tests.

The older `best_material.pt` is preserved as the grayscale FMD baseline. Its 74.45% result belongs to a different dataset/task and is not an NIR score or a direct comparison with this model.

## Windows use

The following commands run from the inner `cv_model_starter` folder, containing `nir_material`. Its Python 3.11 environment is one level above; activation is unnecessary.

```powershell
cd "C:\Users\Dhwanit Kriplani\Downloads\cv_model_starter\cv_model_starter"
..\.venv\Scripts\python.exe .\nir_material\predict_material.py --source "C:\path\sample.png" --output ".\nir_material\outputs\prediction.jpg" --json ".\nir_material\outputs\prediction.json"
```

Or double-click `PREDICT_PHOTO.bat` and paste the photo path, or drag a photo onto it. Both the Python predictor and launchers prefer `best_nir.pt` when present; otherwise they fall back to `best_material.pt`. `--model PATH` explicitly selects another checkpoint.

Optional live demo:

```powershell
.\nir_material\RUN_WEBCAM.bat
.\nir_material\RUN_WEBCAM.bat 1
```

Run one of those commands: the first selects Windows camera 0, the second camera 1. Keep a single material centered. Q/Escape/window-close stops the feed. The square represents the classifier's centre input crop, **not a detected object bounding box**. Camera 0 may be the laptop webcam and does not establish an IR source. A CCTV module may require a compatible video-capture interface; the supplied description does not identify its connector.

On another computer, install Python 3.11 and run `SETUP_WINDOWS.bat`. It locates an existing `.venv` beside the scripts or up to two parents above, or creates a local environment. CPU torch/torchvision and supporting packages are installed. Inference loads the included checkpoint with `weights=None`, so no pretrained weight download occurs at prediction time.

## Reproduce public NIR training

The model is already trained; reproducing it is optional and overwrites `best_nir.pt`.

```powershell
..\.venv\Scripts\python.exe .\nir_material\download_mcubes.py
..\.venv\Scripts\python.exe .\nir_material\prepare_mcubes_crops.py --raw ".\nir_material\mcubes_raw\multimodal_dataset" --output ".\nir_material\mcubes_crops"
..\.venv\Scripts\python.exe .\nir_material\train_nir.py --data ".\nir_material\mcubes_crops" --dataset-name mcubes_nir_material_crops --balanced-loss --sensor-name "FLIR GS3-U3-41C6NIR-C; 750 nm long-pass"
```

`TRAIN_PUBLIC_NIR.bat` performs the same sequence. Approximately 797 MB of NIR images, validity masks, annotations and official split lists are selectively downloaded from the official public archive. ZIP CRCs are checked and verified downloads can be resumed. Raw data/crops are not included in the portable zip; the trained checkpoints, full crop manifest, metrics, code, license and instructions are included.

The original NIR uint16 values use fixed `round(value/257)` full-range conversion to uint8, matching normalization by 65535 in the original loader. The first 192 columns are excluded. White invalid-mask areas are excluded. Crops consist of original unmasked NIR pixels, require at least 90% target material and 99% valid pixels in both the square and the later central model crop, and are limited to four per class/source. No class-shaped input masking or per-image contrast scaling is used. Material IDs are 2/7/15 for MCubeS, unlike FMD's 4/6/9. See `DATASET_SOURCE.md`, `MCUBES_LICENSE.txt` and `mcubes_crop_manifest.json`.

The frozen ImageNet ResNet18 backbone extracts features once; a new three-class head is trained on CPU. `--balanced-loss` applies inverse-frequency weights from training labels only. Validation loss is unweighted and selects the best checkpoint. Exact image hashes and source scene groups are checked; the test is evaluated after selection.

## Later camera samples

First use photo prediction on genuinely new camera samples. For later camera-specific retraining, `COLLECT_IR_SAMPLES.bat` saves labeled frames into `nir_data/SPLIT/CLASS/` only on Space. Select 1=metal, 2=plastic, 3=wood, and Q to stop. Check that the preview is the intended external IR camera. Use different objects and capture sessions for training, validation and test. Every split needs all three classes.

`TRAIN_NIR.bat` trains on those user-supplied camera images. **Back up the public `best_nir.pt` first** because that command replaces it. The generic trainer accepts normalized ordinary 8-bit image frames; high-bit-depth/raw/spectral data requires sensor-specific fixed normalization. It does not silently clip that data.

## Older FMD baseline

Your original train/test parquet files remain in the Downloads project's `nir_material/data/`. Run `TRAIN_FMD.bat` to reproduce the old grayscale baseline, or:

```powershell
..\.venv\Scripts\python.exe .\nir_material\train_material.py --train ".\nir_material\data\train*.parquet" --test ".\nir_material\data\test*.parquet" --output ".\nir_material\best_material.pt"
```

## Prediction interpretation and verification

The model chooses among three known classes. Softmax scores are uncalibrated and are not test accuracy or unknown-material detection. The metal class does not identify aluminium separately. The expected input is a material filling the central crop.

Common model preprocessing is EXIF transpose, grayscale, resize shorter side 256, centre crop 224, grayscale repeated into three channels, and ImageNet normalization. Prediction follows the same grayscale-before-resize order as training. The model checkpoint stores preprocessing, class order and dataset provenance.

Verification covered source/crop alignment, class IDs, scene separation, integral purity calculations, exact source pixels and crop hashes, saved checkpoint loading, CLI photo inference, JSON/image output and Windows photo launcher operation. Both FMD and actual NIR training completed. Live CCTV behavior remains untested.

API reference: [torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
