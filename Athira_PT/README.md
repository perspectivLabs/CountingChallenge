# Counting Challenge Solutions

**Candidate:** Athira PT
**Target:** count the objects in each image and overlay a mask for each one, with > 95 % accuracy.

## 1. Results

Measured against hand-counted ground truth (`ground_truth.csv`, 25 images). Accuracy per image = `100 * (1 - |detected - truth| / truth)`.

| | Non-AI (OpenCV) | AI (fine-tuned YOLOv8-seg) |
|---|---|---|
| **Screws** (7 images, 43 screws each) | **100.0 %** (7/7 above 95 %) | **100.0 %** (7/7 above 95 %) |
| **ScrewAndBolt** (18 images, 10 to 354 objects) | **97.5 %** (15/18 above 95 %) | **92.5 %** (6/18 above 95 %) |

* Both solutions reach the target on **Screws**.
* On **ScrewAndBolt** only the Non-AI solution reaches the target on average (97.5 %). The AI solution averages 92.5 % and is below 95 % on 12 of 18 images, mainly under-counting overlapping screws.
* **Tuning caveat:** the counting parameters of both solutions were tuned on the same 18 ScrewAndBolt images that produce the ScrewAndBolt scores, so those scores are tuned, not held-out results. For the AI model, `Screws/img5` and `img6` were held out of training (both 100 %).

## 2. About the data

`Screws` are clean top-down photos of separate screws. `ScrewAndBolt` contains backlit photos of screws (seen from the side) and **hex nuts**, at different zoom levels, from 10 objects up to piles of 354 touching objects. That is why a simple threshold or a "segment everything" model fails there; see the two READMEs for the details.

## 3. Directory structure

```
Athira_PT/
├── README.md             # this file
├── ground_truth.csv      # hand-counted number of objects per image (used by both evaluators)
├── Non_AI/               # Task 1: classical computer vision
│   ├── count_and_mask.py
│   ├── non_ai_solution.ipynb
│   ├── requirements.txt
│   ├── README.md
│   └── output/           # sample overlays (other generated files are git-ignored)
└── AI/                   # Task 2: AI
    ├── count_and_mask.py # tiled inference + evaluator
    ├── train_yolo.py     # pseudo-label tiling + training
    ├── ai_solution.ipynb
    ├── requirements.txt
    ├── README.md
    └── output/           # sample overlays (other generated files are git-ignored)
```

## 4. Methods in one paragraph each

* **Non-AI** - robust polynomial background fit and threshold give a silhouette. Each connected blob is one object or a group. Piles of nuts are counted by their round holes (one nut per hole, Voronoi split); other groups by `round(blob area / single-object area)` with k-means splitting. Details: `Non_AI/README.md`.
* **AI** - `yolov8n-seg` fine-tuned on pseudo-labels produced by the Non-AI pipeline (no manual labels exist), trained and run on overlapping 640 px tiles at native resolution, with a cross-tile duplicate merge. Training recipe: `AI/README.md` and `AI/train_yolo.py`.

## 5. Quick start

```bash
git clone <your-fork-url>
cd CountingChallenge
python -m venv venv
.\venv\Scripts\activate        # Windows  (source venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```
The datasets go in `data/Screws` and `data/ScrewAndBolt` (not committed).

```bash
# Non-AI
cd Athira_PT/Non_AI
python count_and_mask.py --input ../../data/Screws       --output_dir output/Screws
python count_and_mask.py --input ../../data/ScrewAndBolt --output_dir output/ScrewAndBolt

# AI: train once (about 35 min on a GTX 1660 Ti), then evaluate
cd ../AI
python train_yolo.py --data ../../data --epochs 50 --batch 4
cp runs/screw_seg/weights/best.pt screw_seg.pt
python count_and_mask.py --input ../../data/Screws       --output_dir output/Screws
python count_and_mask.py --input ../../data/ScrewAndBolt --output_dir output/ScrewAndBolt
```

Both scripts print the detected count, the ground truth and the accuracy for every image, and write overlay images and binary masks to the output folder. Trained weights (`*.pt`) are git-ignored, so the AI model must be trained with the commands above.
