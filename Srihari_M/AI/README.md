# AI Counting

## What is here

- `sam3_infference.py`: active batch inference program. It asks SAM3 to find text prompts, segments detections, removes duplicates, and reports counts.
- `AI_Counting.py`: currently contains only a commented note; it is not the runnable inference implementation.
- `Annotation_viewer.py`: GUI for reviewing YOLO box labels and optional JSON segmentation polygons.
- `dataset/`: YOLO-format screw/nut/bolt dataset when downloaded. The current export metadata describes 1,300 images; this checkout contains the `train` split only.
- `SAM3_CHECKPOINT.txt`: Shared Drive URL for the SAM3 checkpoint for inference.
- `sam3.pt`: expected local checkpoint filename. It is ignored by Git and must be obtained separately.

## Setup

Use Python with PyTorch and an Ultralytics version that provides `SAM3SemanticPredictor`. Install the AI implementation's Python packages from the repository root with:

```bash
python3 -m pip install -r Srihari_M/AI/requirements.txt
```

The `requirements.txt` file covers both batch inference and the annotation viewer. PyTorch versions and GPU support depend on the available hardware; install the appropriate PyTorch build for your system if needed.

Place the downloaded model file at `Srihari_M/AI/sam3.pt`. Download/extract the dataset so image files are under `Srihari_M/AI/dataset/train/images`.

## Run inference

The first line of `sam3_infference.py` records the command used during development:

```bash
python3 sam3_infference.py --images ./Sample_data/Sample_data/ --output ./SAM3_inference --model ./sam3.pt --text screw nut --prompt-conf screw=0.35 nut=0.1 --imgsz 1190 --save-binary-masks --overlap 0.2 --nms-iou 0.4
```

Run it from `Srihari_M/AI` only if that sample-image folder exists. It is not included in this repository. For the dataset path in this checkout, run from the repository root:

```bash
python3 Srihari_M/AI/sam3_infference.py --images Srihari_M/AI/dataset/train/images --output Srihari_M/AI/SAM3_inference --model Srihari_M/AI/sam3.pt --text screw nut --prompt-conf screw=0.35 nut=0.1 --imgsz 1190 --save-binary-masks --overlap 0.2 --nms-iou 0.4
```

This example prompts for screws and nuts. Add `bolt` to `--text` to include bolts; adjust the per-prompt confidence values if needed. The example does not enable slicing because it omits `--use-sahi`; `--overlap` only affects sliced inference when slicing is enabled.

## Results to inspect

The output directory contains:

- `BBOX_images/`: input images with detected boxes and count summary.
- `MASK_images/`: colored segmentation overlays.
- `MASK_binary/`: instance-ID mask PNGs when `--save-binary-masks` is supplied.
- `Annotation_JSON_files/`: per-image boxes, labels, confidence values, polygons, and counts.
- `counts_summary.csv`: counts per image and an `ALL_IMAGES` totals row.

Check that every expected object is represented once in the overlay and binary mask, and compare the CSV counts with known labels. A successful run alone does not establish the task's stated accuracy target.

## Review YOLO annotations

The viewer expects an input directory containing `images/` and `labels/` subdirectories. Example for the training split:

```bash
python3 Srihari_M/AI/Annotation_viewer.py Srihari_M/AI/dataset/train
```

Use `A`/left and `D`/right to change images, `L` to record a saved image, `C` to mark accepted, `E` accepted with text, `N` not annotated, `T` not annotated with text, `R` rejected, `M` miscellaneous, and `Esc` to exit. The viewer appends image paths to category `.txt` files in the input directory. Add `--obb` for oriented boxes. For JSON segmentation polygons, use `--seg --json_dir PATH`; the viewer still uses the matching YOLO labels for boxes.