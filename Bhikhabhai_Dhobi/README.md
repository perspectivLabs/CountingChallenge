# CountingChallenge - Bhikhabhai Dhobi

Count the objects (screws, bolts, hex nuts) in each image and overlay a mask per object.

| Folder | Approach | Details |
|---|---|---|
| [`Non_AI/`](Non_AI/README.md) | Classical OpenCV: lighting flattening, Otsu, round-hole nut seeds, edge-aware marker watershed | [README](Non_AI/README.md) |
| [`AI/`](AI/README.md) | YOLO11-seg fine-tuned in Google Colab on masks produced by the Non_AI pipeline | [README](AI/README.md) |

Shared files:

* `ground_truth.csv` - counts for 17 of the 25 images, each with how it was obtained
  (hand count or cell-by-cell audit).
* `evaluate.py` - per-image accuracy `1 - |pred - true| / true`, overall and on the five images
  held out from AI training.

## Setup

Python 3.12. The dataset is not committed: download the two folders from the challenge Google
Drive into `data/` at the repository root:

```
data/ScrewAndBolt_20240713/*.jpg
data/Screws_2024_07_15/*.jpg
```

```bash
cd Bhikhabhai_Dhobi/Non_AI && pip install -r requirements.txt && python count_objects.py
cd ../AI && pip install -r requirements.txt && python count_objects_ai.py   # needs weights/best.pt, see AI/README
cd .. && python evaluate.py Non_AI/output/counts.csv && python evaluate.py AI/output/counts.csv
```
