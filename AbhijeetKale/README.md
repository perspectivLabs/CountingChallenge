# CountingChallenge - Abhijeet Kale

Counts screws/bolts/nuts in each image and overlays a colored mask per object.

## Non_AI (`Non_AI/count_nonai.py`, OpenCV only)
1. Grayscale, flatten uneven lighting (divide by a heavily blurred background).
2. Otsu threshold + morphological open/close, plus an absolute-darkness gate (removes vignette).
3. Connected components; drop specks and lens-vignette blobs at the image edges/corners.
4. Touching objects: `count = round(blob_area / median_single_object_area)`.
5. Colored mask overlay with the count written on the image.

Run: `pip install -r Non_AI/requirements.txt && python Non_AI/count_nonai.py <image_dir> <out_dir>`
Overlays and counts for all provided images are in `Non_AI/results/`.

## AI (`AI/count_sam.py`, Segment Anything)
SAM ViT-B automatic mask generator, then filtering: remove background-size masks, keep only masks darker
than the backdrop, drop masks under 30% of the median area (heads/threads), and de-duplicate overlapping
masks (NMS). Count = remaining masks. No training was done (zero-shot), so there is no training procedure.

Run on a GPU (Colab/Kaggle): `pip install -r AI/requirements.txt && python AI/count_sam.py <image_dir> <out_dir>`
Overlays and counts are in `AI/Results/`.

## Results (Screws_2024_07_15, true count 43 in every image)

| Image | True | Non_AI | AI (SAM) |
|---|---|---|---|
| img1 | 43 | 43 | 42 |
| img1_43_nosy | 43 | 43 | 42 |
| img2 | 43 | 43 | 42 |
| img3 | 43 | 43 | 43 |
| img4 | 43 | 43 | 43 |
| img5 | 43 | 43 | 41 |
| img6 | 43 | 43 | 44 |

Accuracy (1 - mean abs error / true count): Non_AI 100%, AI about 97%.
The true count of 43 was confirmed by a manual count on img3 and by the file name `img1_43_nosy.jpg`.
No ground-truth labels were provided with the data.

## Limitations
- Separated objects (Screws_2024_07_15 and the sparse ScrewAndBolt images) count reliably.
- Dense piles in ScrewAndBolt (hundreds of touching screws/nuts) are approximate for both methods.
  Hidden objects cannot be counted from a top-down 2D image. SAM undercounts touching screws there,
  and the area-ratio estimate in Non_AI is approximate.
