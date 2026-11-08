# Non-AI solution: classical image processing (OpenCV + SciPy + scikit-image)

No model and no training. Every step is deterministic image processing, and the same parameters are used for all 25 images.

## Run

```bash
pip install -r requirements.txt
python count_non_ai.py -i <ScrewAndBolt folder> <Screws folder> -o outputs --save-masks
python ../evaluate.py --pred outputs/counts.csv
```

Outputs:
- `outputs/overlays/*_overlay.jpg`: coloured instance masks, outlines, ids and the total count.
- `outputs/masks/*_labels.png`: 16-bit instance label maps.
- `outputs/counts.csv`: the count for each image.

## Method

| Step | What | Why |
|---|---|---|
| 1 | Resize so the long side is 2000 px | Keeps object size similar across the 12 MP and 1080p photos |
| 2 | Estimate the background with a large grey-scale closing, then subtract it | Removes the strong vignetting (`img3`–`img6`) and lighting gradients |
| 3 | Foreground score = darkness relative to the background + colour distance (Lab) to the backdrop, then an Otsu threshold | Black screws are dark; brass screws are bright but coloured |
| 4 | Find connected components; drop specks and thin strips along the image border | Each component is one item or a cluster of touching items |
| 5 | Estimate the unit area: the largest area *u* such that component areas are close to integer multiples of *u* | A plain median fails when most items touch, or when there are only about 10 items |
| 6a | **Hollow items (nuts):** each round through-hole counts as one nut. Holes are told apart from gaps between touching nuts by size (0.65–1.45 × median hole) and circularity (> 0.85). Masks come from a nearest-hole watershed restricted to nut material. | Hole counting is exact even in piles of 354 touching nuts |
| 6b | **Solid items (screws):** k = round(material area / unit area) per component. Masks come from a watershed seeded at the k thickest points (the heads), falling back to k-means seeds. | Overlapping screws hide little area, so rounding the area ratio is very reliable |
| 7 | Choose the mode automatically: hollow if ≥ 50 % of components contain a valid hole | About 100 % of components in nut images, under 5 % in screw images |

## Results

See [results.md](results.md). On all 25 images: **mean counting accuracy 99.70 %**, worst image 95.92 % (20240713_194206), 21 of 25 counts exact.

## Limitations

- Area-based counting assumes one dominant item size per image. Head-on screws ("discs") are about 1.3 × a lying screw, which causes the +1 to +4 over-counts on the mixed set.
- Masks inside heavily overlapping screw clusters are approximate. The count per cluster is usually right, but the boundaries between screws are guessed by the watershed.
