# AI Dataset

This directory is the expected location for the YOLO screw/nut/bolt dataset. Dataset files are excluded from Git, so obtain and extract them before running inference or viewing annotations.

The included `data.yaml` describes three classes: `bolt`, `nut`, and `screw`. Its paths expect `train/images`, `valid/images`, and `test/images`. The current local export contains `train/` only, so validation and test images are not available in this checkout.

The source metadata is in `README.dataset.txt` and `README.roboflow.txt`. The export is identified as Roboflow's `screw-nut-bolt` dataset, version 1, with a stated total of 1,300 images and a CC BY 4.0 license. Dataset source: [Roboflow Universe](https://universe.roboflow.com/logesh-s-workspace/screw-nut-bolt).

For SAM3 inference, point `--images` at `train/images` (or another folder containing images). For the annotation viewer, pass the split directory, such as `train`, which contains both `images/` and `labels/`.