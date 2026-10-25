# Trained NIR material model

`best_nir.pt` is trained on the actual **NIR_warped** camera channel of [MCubeS](https://github.com/kyotovision-public/multimodal-material-segmentation), using its metal/plastic/wood annotations. It is not grayscale RGB relabeled as NIR.

Measured held-out public NIR crop accuracy: **68.35% (311/455)**. Macro F1: **0.5187**. All 100 head-training epochs completed; validation loss selected epoch 1. This is a first trained model with substantial limitations, especially plastic.

| Material | Correct / test crops | Recall |
| --- | --- | --- |
| Metal | 210 / 250 | 84.00% |
| Plastic | 4 / 57 | 7.02% |
| Wood | 97 / 148 | 65.54% |

Training crops: metal588/plastic50/wood293 (931 total). Validation:186/11/96 (293). Test:250/57/148 (455). Training used inverse-frequency class weights computed only from training labels. Validation loss remained unweighted. The head was fitted on frozen ImageNet ResNet18 features, and test predictions were computed only after the selected checkpoint was saved.

Public source camera: FLIR GS3-U3-41C6NIR-C, 750nm long-pass. The user's separately described 3.6mm IR 1080P CCTV module has **not** supplied sample photos and has **not** been tested. Its camera accuracy remains unknown. These outdoor-scene crop metrics are not a guarantee for CCTV close-ups, a spectral-analysis result, or the original MCubeSNet segmentation benchmark.

The official test source frames were retained. Scene IDs that overlapped test were removed from training/validation; remaining validation scenes were excluded from training. This leaves212/76/102 source frames in disjoint scene groups. Classification crops are raw unmasked NIR pixels, with >=90% material purity and >=99% validity in both the whole square and the later model centre crop. Original uint16 NIR is scaled with round(value/257), then saved as 8-bit. No per-image contrast scaling is used. See `mcubes_crop_manifest.json`, `DATASET_SOURCE.md`, and `best_nir.metrics.json` for provenance/results.

The older `best_material.pt` remains the grayscale FMD baseline (74.45% on a different dataset/test task). Its score is not comparable to this NIR score. Photo/webcam launchers and default Python prediction prefer `best_nir.pt` when present.

Live CCTV testing is the next step. The model forces one of three known classes; scores are uncalibrated and do not constitute an unknown-material detector. Metal is broad and does not separately identify aluminium.
