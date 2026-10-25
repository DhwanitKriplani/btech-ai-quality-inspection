# Integrated material and surface inspection

Start **START_INSPECTION.bat**. This menu runs trained material-region bounding boxes, per-region material classification, and the existing RGB aluminium scratch/corner/side defect segmenter in one application. Photo, video, webcam and separate RGB/NIR input modes are implemented.

The files are also installed in your existing Downloads project under `cv_model_starter/cv_model_starter/integrated_inspection/`. Its existing GPU environment has been configured and verified on the RTX4060. The original CPU environment and original trained models are preserved.

## Use

From the inner cv_model_starter folder:

```powershell
.\integrated_inspection\START_INSPECTION.bat
```

Menu1 RGBphoto,2 RGBlivecamera,3 NIRphoto,4 dualcamera,5 included demonstration images. Or drag a photo onto RUN_PHOTO.bat (RGB) or RUN_NIR_PHOTO.bat (actual normalized NIR). Q/Escape closes the viewer. RUN_WEBCAM.bat optionally accepts an RGBcamera index; RUN_DUAL_CAMERA.bat accepts RGB then NIRindices, default0 and1.

Direct commands from that same inner project folder:

```powershell
..\.venv_gpu\Scripts\python.exe .\integrated_inspection\run_inspection.py --source "C:\photos\sample.jpg" --output ".\integrated_inspection\outputs\photo" --show
..\.venv_gpu\Scripts\python.exe .\integrated_inspection\run_inspection.py --source "C:\photos\sample_ir.png" --input-mode nir --output ".\integrated_inspection\outputs\nir_photo" --show
..\.venv_gpu\Scripts\python.exe .\integrated_inspection\run_inspection.py --source 0 --nir-source 1 --output ".\integrated_inspection\outputs\dual" --show
```

For video, pass a video filename as --source. A photo produces annotatedJPEG+JSON; video/camera produces MP4+per-frameJSONL+summaryJSON. Reports store original pixel coordinates, masks, detector/classifier estimates separately, componentmodel hashes/domains, and display transforms. `--max-frames N` limits video/camera processing; `--stride N` samples frames. Use `--device cpu` to force CPU, or0 for CUDA. Python3.11, ultralytics8.4.184 and supporting packages are listed in requirements.txt. SETUP_WINDOWS.bat provides a portable CPU setup; the GPUenvironment on this PC is already ready.

## How stages work together

- **RGB:** learned RGB material-region boxes are classified with the saved FMD classifier. The aluminium defect model runs on the same originalRGB frame and draws real defect boxes/masks. A defect is associated with the smallest same-frame material-region box containing its centre; otherwise it stays unassigned. This is geometric association, not proof of material or a whole-object identity.
- **NIR:** the separately trained NIRregion detector and saved NIRclassifier run; RGBdefect weights are not applied to NIRframes.
- **Dual:** both pipelines run in one application, in separate panes. Inputs are read sequentially and are not hardware-synchronized or spatially registered. Boxes are not transferred between cameras without calibration. Physicalcamera alignment is still a later hardware task.

Boxes come from actual model outputs. There is no fixed centre square masquerading as detection. Material boxes represent connected surface/material regions, not guaranteed whole-object instances. Detector and auxiliary classifier estimates can disagree; scores are uncalibrated and are not multiplied into a fake combined confidence. The displayed material check is unsure on disagreement; JSON preserves all raw estimates.

## Actual measured results

The new material-region detectors are **weak experimental models**, especially plastic. The integration is functional; material boxes/classification are not reliable enough for automatic sorting decisions and require labeled target-camera data for improvement.

| Component | Test domain | mAP50 | mAP50-95 |
| --- | --- | --- | --- |
| RGBmaterial boxes |102 publicMCubeS testframes|11.76%|5.48%|
| NIRmaterial boxes |102 publicMCubeS testframes|9.86%|4.90%|
| Aluminium defect boxes |50 user-dataset RGBtestimages, all4classes|99.23%|77.54%|
| Aluminium defect masks |same50images, all4classes|98.11%|57.04%|

Scratch-only boxAP50 is98.42%; maskAP50 is97.12% on that same aluminium test set. All-class defect averages include Black Marker, but the app excludes the marker as a physical defect bydefault (`--include-markers` shows it separately). The scratch/defect model has not been validated on plastic, wood or NIR.

The older standalone crop classifiers measured74.45% grayscaleFMD accuracy and68.35% publicNIRcrop accuracy; these are different tests/metrics from boxmAP and are not whole-system accuracy. NIRcrop plastic recall is4/57. No end-to-end accuracy for a paired RGB/NIR productioncamera setup is claimed. No detection means no defect was detected, not a guarantee of a defect-free item.

## Provenance and reproducibility

New detectors use actual [MCubeS](https://github.com/kyotovision-public/multimodal-material-segmentation) RGB and NIRchannels, their materialGT and invalid masks. Same scene-purged split sourceframes212train/76val/102test as previousNIR work; all originaltest sourceIDs are kept, with shared scene prefixes purged from train/validation. MaterialIDs2/7/15 map to metal/plastic/wood0/1/2. Unmasked sourceframes use fixed16-bit to8-bit `/257` scaling and the original left192column crop. Boxes are derived from valid8-connected labelcomponents with area>=256pixels, side>=12pixels and fill>=.10. No class-shaped input masking or test-set tuning is used. Each YOLO26nano detector trained up to40epochs and selected best weights using validation, then evaluated once on the test split.

The pretrained scratch segmenter was verified as an actual custom-trained4-class YOLO26segmentation model. Its original335/95/50 train/validation/test files and originalcheckpoint were preserved; the held-out evaluation checked image/label hashes before and after.

TRAIN_DETECTORS.bat optionally reproduces materialdetector training. It downloads roughly3.5GB of selected publicarchive data, prepares labels, and trains/evaluates both detector modalities. Datasetpreparation helpers and the original MCubeS MITlicense/citation are in source_tools. Raw datasets, trainingcaches, unrelated testfixtures and the Python environments are not included in the portablezip. Trainingoutputs remain on this PC.

## Verification and next step

Nine focused tests cover coordinate association, no fabricated boxes, correct NIR stage selection, markerhandling, paired coordinate systems, high-depth rejection, and photo/video fileoutput. Actual-weight photo/video checks exercise model loading, CUDA inference, all5component models, real boxes/masks/classifier scores and savedJPEG/MP4/JSONoutputs. Included examples are clearly identified TRAINING images for software demonstrations, not independent accuracy tests.

**The user's CCTV camera has not yet supplied samples and was not opened during these checks.** Windows previously reported only Integrated Webcam. Once the CCTV feed is available through a suitable interface, test actualsamples and collect labeledmaterial boundingboxes/defects from the target setup. Separately registered RGB/NIR cameras require calibration before cross-sensor object association.
