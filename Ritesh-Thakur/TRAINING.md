# Fine-tuning YOLOv8 on Custom Screw/Bolt Data

## When to fine-tune?
The pretrained `yolov8m.pt` (COCO-80 classes) works reasonably well for
generic objects.  If you need **>95% accuracy** on your specific screw dataset,
fine-tuning on labelled images is recommended.

---

## 1 — Label your images

Use **[Roboflow](https://roboflow.com)** or **[CVAT](https://cvat.ai)** (both free):

1. Upload the images from `ScrewAndBolt_20240713` and `Screws_2024_07_15`.
2. Draw bounding boxes around every screw/bolt.
3. Export in **YOLOv8 format** → you get a `dataset.yaml` + `images/` + `labels/` folders.

---

## 2 — dataset.yaml structure

```yaml
path: /content/drive/MyDrive/CountingChallenge/dataset   # root
train: images/train
val:   images/val
test:  images/test   # optional

nc: 1               # number of classes
names: ['screw']    # class names
```

---

## 3 — Fine-tune

```python
from ultralytics import YOLO

model = YOLO('yolov8m.pt')   # start from pretrained
model.train(
    data     = 'dataset.yaml',
    epochs   = 60,
    imgsz    = 640,
    batch    = 16,
    name     = 'screw_counter',
    patience = 15,       # early stopping
    augment  = True,     # mosaic, flip, HSV jitter, etc.
    cache    = True,     # cache images in RAM for speed
)
```

Best weights → `runs/detect/screw_counter/weights/best.pt`

---

## 4 — Use fine-tuned weights

```bash
# CLI
python count_objects_ai.py /path/to/images \
    --yolo-weights runs/detect/screw_counter/weights/best.pt \
    --conf 0.30
```

```python
# Python
from count_objects_ai import process_directory
results = process_directory(
    'data/', 'AI_output/',
    yolo_weights='runs/detect/screw_counter/weights/best.pt',
    conf=0.30
)
```

---

## 5 — Expected accuracy

| Stage                        | Approx. accuracy |
|------------------------------|-----------------|
| Pretrained YOLOv8m (COCO)   | 80–90%          |
| Fine-tuned on 100+ images    | 95–98%          |
| Fine-tuned + data augment    | 97–99%          |

> Rule of thumb: **~50 labelled images per class** gives a solid baseline.
> **200+** images give near-perfect accuracy on screws/bolts.

---

## 6 — Data augmentation tips (built into Ultralytics)

Enabled by default with `augment=True`:
- Mosaic (4-image composite)
- Random horizontal/vertical flip
- HSV colour jitter
- Random scale, translate, shear
- Copy-paste augmentation

These are especially useful when your dataset is small (<200 images).
