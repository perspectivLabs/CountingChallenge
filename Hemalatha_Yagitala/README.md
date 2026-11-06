# Hemalatha Yagitala: Counting Challenge

Count the items in each image and overlay a mask on each one (target: accuracy > 95%).

| Folder | Approach | Mean count accuracy |
|---|---|---|
| `Non_AI/` | Classical OpenCV: flat-field correction, hole-seeded nut counting, area-based screw counting with upright-screw cores, orientation-aware k-means masks | **99.61%** (all 25 images) |
| `AI/` | YOLO11n-seg trained on 640 px tiles with model-assisted labels, tiled inference with mask-level duplicate removal | **99.00%** (8 held-out test images) |

Accuracy per image = 1 − |predicted − actual| / actual, against hand-counted ground truth in
`ground_truth.csv`. Each folder's README has the per-image results, method, and how to run it.

Data isn't committed. Download it from the link in the challenge README and extract it into `data/`.
