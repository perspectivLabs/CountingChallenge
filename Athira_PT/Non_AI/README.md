# Non-AI Solution: Object Counting & Mask Overlay (OpenCV)

**Candidate:** Athira PT
**Task:** count the objects in an image and overlay a mask for each, with > 95 % accuracy, using classical computer vision only.

## Results (against hand-counted ground truth, `../ground_truth.csv`)

| Dataset | Images | Mean accuracy | Images > 95 % |
|---|---|---|---|
| Screws (43 screws per image) | 7 | **100.0 %** | 7 / 7 |
| ScrewAndBolt (10 to 354 objects: screws + hex nuts) | 18 | **97.5 %** | 15 / 18 |

Accuracy per image = `100 * (1 - |detected - truth| / truth)`.

Images below 95 %: `20240713_192951` (24 vs 21), `20240713_193135` (17 vs 21), `20240713_194200` (104 vs 98). All three contain screws lying across each other; area-based counting cannot reliably tell two overlapping screws from one large one.

> **Tuning caveat.** The counting parameters (`SINGLES_FRACTION = 0.65`, `OVERLAP_GAIN = 1.0`) and the nut-hole circularity cutoff (0.7) were tuned on the same 18 ScrewAndBolt images that are used for the score above, so that number is a tuned result, not a held-out one. The Screws result needed no tuning.

## Why ScrewAndBolt is hard

The images are backlit silhouettes, taken at different zoom levels. They contain screws seen from the side, hex nuts, and **piles of hundreds of touching nuts**. Two things broke the earlier version:

1. A *local* background estimate (median filter) treats a big pile as background, so the pile disappears.
2. Counting by distance-transform peaks and watershed splits one screw (wide head + thin shank) into two.

## Pipeline (`count_and_mask.py`)

1. **Silhouette** - robust background surface: a degree-6 polynomial in (x, y) fitted on a 20x downscaled image, refitted 8 times using only pixels *not clearly darker than the fit*. It follows vignetting but is never pulled down by objects. Foreground = `background - gray`, thresholded at `max(25, 0.6 * Otsu)`.
2. **Connected components** - every blob is one object or a group of touching objects.
3. **Single-object area** - median area of the smallest 65 % of blobs (isolated objects).
4. **Count per blob**
   * Blob has >= 2 round holes of consistent size (circularity > 0.7, size within 0.5x-2x of the median hole) -> it is a pile of nuts: **one nut per hole**, pixels assigned to the nearest hole (Voronoi split).
   * Otherwise `k = round(blob_area / single_area)`, and a blob with `k > 1` is split into `k` masks with k-means on its pixel coordinates.
5. **Noise rejection** - blobs below 45 % of a single object, or below 60 % if they touch the image border (dark vignette corners), are dropped.
6. **Output** - translucent coloured masks, contours, centroids and `#id` labels, plus a banner with the total.

Images are processed at full resolution (no downscaling) because the objects can be small.

## Run

```bash
pip install -r requirements.txt
python count_and_mask.py --input ../../data/Screws       --output_dir output/Screws
python count_and_mask.py --input ../../data/ScrewAndBolt --output_dir output/ScrewAndBolt
python count_and_mask.py --input path/to/one_image.jpg                    # single image
```

The evaluator reads `../ground_truth.csv` and prints the detected count, the true count and the accuracy for every image. The notebook `non_ai_solution.ipynb` runs the same code with figures. Set `DATA_DIR` if the dataset is not in `<repo>/data`.

## Files

| File | Purpose |
|---|---|
| `count_and_mask.py` | pipeline, per-image function `detect_items_non_ai`, evaluator and CLI |
| `non_ai_solution.ipynb` | notebook walkthrough and benchmark |
| `requirements.txt` | dependencies |
| `output/` | generated overlays and masks (git-ignored except `.gitkeep`) |
