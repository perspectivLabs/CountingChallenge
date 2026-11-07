# Non_AI solution: classical computer vision (OpenCV / scikit-image)

Run: `python run_non_ai.py --input <folder with the images> --output <out folder>` (searches sub-folders for `.jpg`).
Requirements: [`requirements.txt`](requirements.txt) (opencv-python-headless, numpy, scipy, scikit-image). Python 3.11.

Outputs per image: `<name>.jpg` overlay (one colour per counted item, count printed), `<name>_labels.png` (16-bit instance
map) and one `counts.csv`. The overlays and counts from the final run are in [`results/`](results/).

## Method
Every image is first classified as a *nut scene* or a *screw scene* automatically: if at least 100 round see-through holes
are found it is a nut scene (the nut photos have about 354; every other photo has fewer than 40). The 25 photos fall into
two kinds of scene, and each gets its own pipeline.

**Nut scenes** (`nut_holes.py`, `nut_masks.py`)
1. Estimate the background with a large max-filter on a 1/8-size copy plus a heavy blur (a local median fails inside
   big piles) and take the foreground where `gray / background < 0.72`.
2. Every hex nut has exactly one round hole. Enclosed holes (`fill_holes(mask) & ~mask`) are kept if they are round
   (circularity >= 0.85) and of the dominant size. **The number of holes is the count.**
3. Each nut's mask is the foreground pixels nearest to its hole (a Voronoi assignment), capped at one nut radius that is
   measured from isolated nuts.

**Screw scenes** (`count_classical.py`)
1. Resize to 1600 px on the long side, grayscale, blur; background = large median blur, `diff = background - gray`.
2. Otsu threshold (with a floor of 25), open/close, fill holes.
3. Split touching screws: distance transform, smoothed, h-maxima (h = 1.4) as markers, marker-controlled watershed,
   fragments below 0.35 of a typical item merged into the neighbour they share the longest border with.
4. A robust single-item area is measured from the segments themselves (starting from a low percentile, because merged
   pairs would inflate the median). Any segment 1.7x larger than that is split into `round(area / reference)` parts with
   k-means on its pixel coordinates.
5. Fragments smaller than 0.3 of the reference are dropped. The count is the number of remaining segments.

## Accuracy definition
Per image `1 - |pred - true| / true`, averaged, plus the exact-match rate. Counts and overlays from the final run are in
`results/` (`results/counts.csv`, `results/overlays/`).

## Results (all 25 photos)
### Accuracy against the final ground truth
The final reference count for every photo comes from the tile labels: each 1024 px tile was checked and corrected by hand
in an annotation tool, the tiles were stitched back onto the full photo and an item seen in two overlapping tiles was
counted once. Many of those labels started from this Non_AI pipeline's masks and were then corrected by hand, so this is
**not fully independent** of the method being scored; it is still a hand-checked count.

| Group | Photos | Non_AI | Ground truth | Accuracy |
|---|---|---|---|---|
| Sparse screws | 192951, 193135, 193650, 193659 | 21, 22, 10, 10 | 21, 22, 10, 10 | 100% |
| Gold screws | img1..img6 + noisy | 43 x7 | 43 x7 | 100% |
| Nuts | 194541, 194551, 194606, 194621, 194630 | 354, 354, 353, 353, 354 | 354, 354, 356, 353, 354 | 99.2-100% |
| Medium density | 194200, 194206, 194215, 194232, 194256, 194316 | 103, 103, 100, 99, 99, 98 | 96, 98, 98, 100, 97, 99 | 92.7-99.0% |
| Dense piles | 193831, 193839, 193907 | 204, 209, 211 | 193, 197, 196 | 92.3-94.3% |

**Mean accuracy over all 25 photos: 98.4%** (15 of 25 exact). Five photos are below 95%: the three dense piles and two
medium photos (194200: 92.7%, 194206: 94.9%). In those photos the method over-splits touching screws, giving 5-8% too many.

## Known limitations
- Heavily overlapping screws in the dense piles are over-split (5-8% too many against the ground truth); this is the weakest case.
- A nut with no detectable hole (for example a very dark one) is missed.
- The k-means step can occasionally create a small extra region in a gap between three screws.
- Counts come from a fixed set of thresholds tuned by eye on these 25 photos; other lighting would need re-tuning.
