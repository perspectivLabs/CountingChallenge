# Counting Challenge - Chandrahaas Madala

Counts items (screws / washers) in each image and overlays instance masks.

## Run
```
# Non-AI (OpenCV)
cd Non_AI && pip install -r requirements.txt
python count_screws.py --input <dataset_dir> --output output

# AI (pretrained FastSAM, zero-shot, no training)
cd AI && pip install -r requirements.txt
# FastSAM-s.pt is auto-downloaded by ultralytics (or place it in AI/)
python count_screws_ai.py --input <dataset_dir> --output output
```
Outputs: `<image>_overlay.jpg` (coloured masks + count) and `counts.csv`. `samples/` has a few example overlays.

## Method
**Non-AI:** background estimation (morphological close + blur) -> background-subtracted darkness map -> Otsu (relaxed) threshold -> morphology -> connected components. Blobs about one item in area count as 1; larger (touching) blobs are split with k-means into k = round(area / single-item area) pieces, where single-item area is the median of the smallest third of blobs.

**AI:** pretrained FastSAM (class-agnostic segmentation, no training or fine-tuning). The image is tiled (800 px tiles, 640 stride) so tiny items are resolved; masks are filtered by size and contrast with local background, cut-off masks at tile borders are dropped, duplicates removed. Because SAM-style models merge touching items, masks that are about k x the single-item area are counted as k (capped at 4).

## Results (counts per image)
| Image | Non-AI | AI |
|---|---|---|
| ScrewAndBolt_20240713/20240713_192951.jpg | 21 | 20 |
| ScrewAndBolt_20240713/20240713_193135.jpg | 21 | 22 |
| ScrewAndBolt_20240713/20240713_193650.jpg | 10 | 9 |
| ScrewAndBolt_20240713/20240713_193659.jpg | 10 | 9 |
| ScrewAndBolt_20240713/20240713_193831.jpg | 201 | 193 |
| ScrewAndBolt_20240713/20240713_193839.jpg | 202 | 186 |
| ScrewAndBolt_20240713/20240713_193907.jpg | 196 | 181 |
| ScrewAndBolt_20240713/20240713_194200.jpg | 112 | 95 |
| ScrewAndBolt_20240713/20240713_194206.jpg | 110 | 95 |
| ScrewAndBolt_20240713/20240713_194215.jpg | 105 | 96 |
| ScrewAndBolt_20240713/20240713_194232.jpg | 106 | 100 |
| ScrewAndBolt_20240713/20240713_194256.jpg | 102 | 88 |
| ScrewAndBolt_20240713/20240713_194316.jpg | 101 | 94 |
| ScrewAndBolt_20240713/20240713_194541.jpg | 365 | 307 |
| ScrewAndBolt_20240713/20240713_194551.jpg | 371 | 304 |
| ScrewAndBolt_20240713/20240713_194606.jpg | 361 | 342 |
| ScrewAndBolt_20240713/20240713_194621.jpg | 175 | 346 |
| ScrewAndBolt_20240713/20240713_194630.jpg | 361 | 351 |
| Screws_2024_07_15/img1.jpg | 43 | 42 |
| Screws_2024_07_15/img1_43_nosy.jpg | 43 | 42 |
| Screws_2024_07_15/img2.jpg | 43 | 43 |
| Screws_2024_07_15/img3.jpg | 44 | 43 |
| Screws_2024_07_15/img4.jpg | 44 | 43 |
| Screws_2024_07_15/img5.jpg | 44 | 43 |
| Screws_2024_07_15/img6.jpg | 44 | 43 |

## Caveat on accuracy
No ground-truth counts were provided, except the filename `img1_43_nosy.jpg` (43 items; Non-AI: 43, AI: 42). The two methods agree within a few percent on sparse images and the 43-screw set, but differ substantially on dense piles (e.g. `20240713_194621.jpg`: 175 vs 346; visual inspection suggests the AI count is closer there). The 95% target has therefore **not** been verified; please validate against manual counts.
