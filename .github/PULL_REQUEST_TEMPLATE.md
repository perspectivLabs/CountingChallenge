## Info

### Name
<!-- Please enter your name as per submitted resume. -->
Athira PT

### Python Version
<!-- Please specify the Python version you used (e.g., Python 3.8). -->
Python 3.12.3

## Description

### AI
<!-- Provide a brief summary on how the solution was derived using AI Technique -->
Fine-tuned a single-class `yolov8n-seg` (COCO-pretrained) instance-segmentation model. No manual labels exist, so the training masks are **pseudo-labels** generated automatically by the Non_AI pipeline (self-training). Because the objects are small (80-250 px in a 3000 x 4000 photo), training and inference use **overlapping 640 x 640 tiles at native resolution**. Inference merges cross-tile duplicates and the count is the number of remaining masks.

* Training recipe: `Athira_PT/AI/train_yolo.py` (347 train / 27 val tiles, 50 epochs, batch 4, `imgsz=640`, scale/flip/rotate/mosaic augmentation, about 35 min on a GTX 1660 Ti). `Screws/img5` and `img6` are held out from training. Details: `Athira_PT/AI/README.md`.
* An earlier zero-shot FastSAM version was dropped because it split every screw into head and shank and produced duplicate masks.
* Result against hand-counted ground truth (`Athira_PT/ground_truth.csv`): **Screws 100.0 %** (7/7 images above 95 %, including the held-out ones), **ScrewAndBolt 92.5 %** mean (6/18 images above 95 %). The AI solution does **not** reach 95 % on ScrewAndBolt; it under-counts overlapping screws.
* The trained weights (`*.pt`) are git-ignored; run `train_yolo.py` to recreate them (commands in the README).

### Non_AI
<!-- Provide a brief summary on how the solution was derived using Non_AI Technique -->
Classical OpenCV pipeline at full resolution:

1. Silhouette from a robust degree-6 polynomial background fit (follows vignetting, is not pulled down by large piles) and an Otsu-scaled threshold.
2. Connected components: each blob is one object or a group of touching objects.
3. Piles of hex nuts are counted by their round holes (one nut per hole, Voronoi split). Other groups are counted as `round(blob area / single-object area)` and split with k-means.
4. Tiny blobs and small border-touching blobs (dark vignette corners) are rejected; translucent masks, contours and ids are overlaid.

* Result against hand-counted ground truth: **Screws 100.0 %** (7/7 images above 95 %), **ScrewAndBolt 97.5 %** mean (15/18 images above 95 %). Weak images: `192951`, `193135`, `194200` (screws lying across each other). Details: `Athira_PT/Non_AI/README.md`.

## Additional Comments
<!-- Optional: Add any additional comments or context about your changes here. -->
* Accuracy per image = `100 * (1 - |detected - truth| / truth)`. The ground truth is my own hand count of all 25 images (`Athira_PT/ground_truth.csv`); both evaluators read it.
* The counting parameters of both solutions were tuned on the same 18 ScrewAndBolt images that produce the ScrewAndBolt scores, so those scores are tuned, not held-out results.
* Notebooks: `Athira_PT/Non_AI/non_ai_solution.ipynb` and `Athira_PT/AI/ai_solution.ipynb` (committed without outputs; both were run end to end and print the results above).
* Requirements files are in `Athira_PT/Non_AI/`, `Athira_PT/AI/` and the repository root. Datasets are not included.
