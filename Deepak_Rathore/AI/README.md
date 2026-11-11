# CountingChallenge — YOLO11n Instance Segmentation (Dataset V3)

## Project overview

This README documents the V3 dataset preparation and YOLO11n instance-segmentation training run for the **Screws & Nuts** counting task.

The model predicts two classes:

| Class ID | Class name |
|---:|---|
| 0 | Nuts |
| 1 | Screws |

The goal is to detect each object, produce an instance mask for each detection, and count nuts and screws in an image.

## Dataset V3

- **Roboflow workspace:** `deepak-rathore-2rbjk`
- **Roboflow project:** `screws-nuts-resfz` (`Screws & Nuts`)
- **Dataset version:** `3`
- **Export format:** YOLO segmentation (`yolov8`)
- **Colab directory:** `/content/Screws-&-Nuts-3`
- **Dataset YAML:** `/content/Screws-&-Nuts-3/data.yaml`

### Dataset split

| Split | Images | Labels |
|---|---:|---:|
| Train | 67 | 67 |
| Validation | 3 | 3 |
| Test | 2 | 2 |
| **Total** | **72** | **72** |

### Annotation audit

| Annotation type | Rows |
|---|---:|
| Polygon annotations | 2,887 |
| Bounding-box-only annotations | 0 |
| Invalid annotations | 0 |

The audit passed. All scanned training and validation images were reported as non-corrupt by Ultralytics. V3 contains polygon annotations suitable for instance segmentation; unlike the earlier V1 export, this V3 export had no box-only rows.

**Dataset limitation:** validation and test splits are very small (3 and 2 images). Their metrics are preliminary and cannot establish robust real-world performance. Use a larger independent benchmark for final counting evaluation.

## Training environment

| Item | Value |
|---|---|
| Framework | Ultralytics 8.4.210 |
| Python | 3.13.16 |
| PyTorch | 2.14.1+cu130 |
| GPU | NVIDIA Tesla T4 (14,913 MiB reported memory) |
| Model | YOLO11n-seg, pretrained from `yolo11n-seg.pt` |
| Task | Instance segmentation |
| Classes | 2 |
| Parameters | Approximately 2.84 million |

The run transferred 555 of 561 items from the pretrained weights.

## Training configuration

| Parameter | Value |
|---|---:|
| Epochs requested | 100 |
| Early-stopping patience | 20 |
| Epochs completed | 52 |
| Best epoch | 32 |
| Image size | 1024 |
| Batch size | 4 |
| Optimizer | AdamW |
| Initial learning rate (`lr0`) | 0.001 |
| Cosine learning-rate schedule | Enabled |
| Maximum detections (`max_det`) | 400 |
| Mosaic | 0.2 |
| Close mosaic | 10 |
| Horizontal flip (`fliplr`) | 0.5 |
| Vertical flip (`flipud`) | 0.5 |
| Rotation (`degrees`) | 10 |
| Mixed precision (`amp`) | Enabled |
| Random seed | 42 |
| Data-loader workers | 2 |

Training stopped after epoch 52 because validation performance had not improved for 20 epochs. The best checkpoint was saved from epoch 32.

## Reproducing the training run

### 1. Install dependencies

```python
!pip install -q roboflow ultralytics pyyaml
```

### 2. Mount Google Drive (optional)

```python
from google.colab import drive
drive.mount("/content/drive")
```

### 3. Download Roboflow Dataset V3

Enter a regenerated Roboflow API key when prompted. Do not hard-code API keys in notebooks or commit them to Git.

```python
from roboflow import Roboflow
from getpass import getpass
from pathlib import Path

api_key = getpass("Enter your Roboflow API key: ")
rf = Roboflow(api_key=api_key)
project = rf.workspace("deepak-rathore-2rbjk").project("screws-nuts-resfz")
version = project.version(3)
dataset = version.download("yolov8")

DATASET_DIR = Path(dataset.location)
print("Dataset:", DATASET_DIR)
print("YAML:", DATASET_DIR / "data.yaml")
```

### 4. Train YOLO11n-seg

```python
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")
results = model.train(
    data=str(DATASET_DIR / "data.yaml"),
    epochs=100,
    patience=20,
    imgsz=1024,
    batch=4,
    device=0,
    workers=2,
    optimizer="AdamW",
    lr0=0.001,
    cos_lr=True,
    max_det=400,
    mosaic=0.2,
    close_mosaic=10,
    fliplr=0.5,
    flipud=0.5,
    degrees=10,
    amp=True,
    seed=42,
    project="/content/countingchallenge_runs",
    name="yolo11n_seg_v3",
    save=True,
    plots=True,
)
```

### 5. Validate the best checkpoint

```python
best_model = YOLO(
    "/content/countingchallenge_runs/yolo11n_seg_v3/weights/best.pt"
)
val_metrics = best_model.val(
    data=str(DATASET_DIR / "data.yaml"),
    split="val",
    imgsz=1024,
    batch=4,
    max_det=400,
)
```

### 6. Evaluate on the test split

```python
test_metrics = best_model.val(
    data=str(DATASET_DIR / "data.yaml"),
    split="test",
    imgsz=1024,
    batch=4,
    max_det=400,
)
```

## V3 results

### Best-checkpoint validation

| Metric | Box | Mask |
|---|---:|---:|
| Precision | 0.959 | 0.959 |
| Recall | 1.000 | 1.000 |
| mAP@50 | 0.985 | 0.985 |
| mAP@50–95 | 0.882 | 0.821 |

Per-class mask results:

| Class | Validation images | Instances | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|---:|---:|
| Nuts | 1 | 5 | 0.954 | 1.000 | 0.995 | 0.895 |
| Screws | 2 | 83 | 0.965 | 1.000 | 0.976 | 0.746 |

### Test split

The test evaluation contained 2 images and 31 instances.

| Metric | Box | Mask |
|---|---:|---:|
| Precision | 0.956 | 0.956 |
| Recall | 0.752 | 0.752 |
| mAP@50 | 0.881 | 0.881 |
| mAP@50–95 | 0.719 | 0.740 |

Per-class test results:

| Class | Test images | Instances | Precision | Recall | mAP@50 | Mask mAP@50–95 |
|---|---:|---:|---:|---:|---:|---:|
| Nuts | 1 | 10 | 0.926 | 0.600 | 0.791 | 0.655 |
| Screws | 1 | 21 | 0.986 | 0.905 | 0.970 | 0.825 |

**Interpretation:** validation metrics are strong, but test recall is lower, particularly for Nuts. Given the tiny splits, these results are preliminary.

## Checkpoint and output locations

The run output directory was:

```text
/content/countingchallenge_runs/yolo11n_seg_v3/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.csv
├── results.png
├── confusion_matrix.png
├── labels.jpg
└── ...
```

Use `weights/best.pt` for inference. If the notebook's Drive-copy cell was run, the intended destinations were:

```text
/content/drive/MyDrive/CountingChallenge/models/yolo11n_seg_v3_best.pt
/content/drive/MyDrive/CountingChallenge/models/yolo11n_seg_v3_last.pt
/content/drive/MyDrive/CountingChallenge/runs/yolo11n_seg_v3/
```

## Loading the trained model for inference

Load the segmentation checkpoint directly with Ultralytics. Do not initialize `yolo11n.pt` and manually inject checkpoint weights.

```python
from ultralytics import YOLO

model = YOLO("yolo11n_seg_v3_best.pt")
results = model.predict(
    source="path/to/image.jpg",
    imgsz=1024,
    conf=0.60,
    iou=0.50,
    max_det=400,
    save=True,
)
```

Class names are `Nuts` and `Screws`. Count detections by class ID in each prediction result. The `conf=0.60` threshold was used for visual/counting tests; it has not been established as the globally optimal threshold. Select the final threshold using a representative held-out benchmark.

## Evaluation notes and next steps

- Detection/segmentation mAP is not the same as exact object-count accuracy.
- High validation mAP on three images does not demonstrate generalization.
- V3 still needs to be evaluated on the same independent external benchmark used for V2 before claiming improved real-world counting performance.
- Report class-wise count MAE, total-count MAE, and exact-count accuracy alongside precision, recall, and mAP.
- Inspect false positives, missed objects, and class confusion in annotated predictions.
- Keep the test set separate from training and threshold tuning.

## References

- Assignment repository: https://github.com/AlgoDr/CountingChallenge
- Original dataset folder: https://drive.google.com/drive/folders/1TuM4CgGI3WBqOHNFjuzjaEzLOH5Yw43_
- Roboflow project: https://app.roboflow.com/deepak-rathore-2rbjk/screws-nuts-resfz

---

*Documentation reflects the V3 training run and outputs available on 11 November 2026. V3 external-benchmark results were not yet available when this README was prepared.*
