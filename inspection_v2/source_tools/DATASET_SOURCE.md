# Public NIR training source

This module uses the **NIR_warped** image channel from MCubeS, with its material annotations (**GT**) and invalid-region masks (**NIR_warped_mask**). It does not relabel RGB or polarization images as NIR.

- Official project: https://vision.ist.i.kyoto-u.ac.jp/research/mcubes/
- Official repository and download link: https://github.com/kyotovision-public/multimodal-material-segmentation
- Official dataset archive ID: `14egTCyC0Pampb7imrXVwaDRffHN7FZxh`
- Paper: https://vision.ist.i.kyoto-u.ac.jp/pubs/YLiang_CVPR22.pdf
- Authors: Yupeng Liang, Ryosuke Wakaki, Shohei Nobuhara and Ko Nishino.
- Paper: *Multimodal Material Segmentation*, CVPR 2022, pages 19800–19808.
- Original camera described in the paper: FLIR GS3-U3-41C6NIR-C with a 750 nm long-pass filter.
- Repository license: MIT, Copyright 2022 Yupeng Liang; included as `MCUBES_LICENSE.txt`.

`download_mcubes.py` fetches and checks ZIP CRC values for the NIR images, masks, annotations and split lists. It uses public HTTP byte-range requests to select approximately 797 MB from the complete 9.27 GB archive. No authentication is bypassed and no private data is accessed.

The official class indices for this source are **2=metal, 7=plastic, 15=wood**. These differ from FMD's 4/6/9 indices. Original NIR values are uint16 and are mapped to 8-bit with one fixed full-range scaling rule, never per-image contrast scaling. The first 192 image columns are excluded, matching the official loader. White invalid-mask pixels are excluded.

Classification examples are unmasked squares selected using annotation purity; the NIR input is not replaced with ground-truth label shapes. The crop manifest records source image IDs, split, coordinates, purity and conversion. Split membership remains derived from the official lists. Because some source scene prefixes occur in more than one official split, source scenes present in the test split are purged from validation/training; training also excludes the remaining validation scenes. Original official test images are kept, and no test frames are moved into training.

Metrics from this derived patch-classification task are not MCubeSNet semantic-segmentation benchmark scores. They do not establish accuracy on the user's separately described 3.6 mm IR CCTV module; its samples must be evaluated later.
