# Non-AI solution: classical OpenCV counter

`count_objects.py` counts screws, bolts and hex nuts and draws one coloured mask per object.
No learned parameters, about 1-2 s per image on a laptop CPU.

```bash
pip install -r requirements.txt
python count_objects.py --data ../../data --out output    # data/ = the two dataset folders
python ../evaluate.py output/counts.csv
```

Outputs: `output/<image>_overlay.jpg` (masks), `output/<image>_regions.npy` (label image, also used
as training labels for the AI solution) and `output/counts.csv`.

## How it works

| Step | What | Why |
|---|---|---|
| 1. Working size + lighting | Resize to a 2000 px long side; divide the LAB lightness by a heavily smoothed, morphologically closed background | The two folders differ in resolution (1080p vs 4000 px) and have uneven light and dark lens corners |
| 2. Foreground | Otsu threshold on the flattened image; drop specks < 0.3 of an object and blobs that run along the frame (vignette) | One threshold works for all 25 images after flattening |
| 3. Nut holes | Enclosed bright regions, hole-sized, **circularity >= 0.8**; used only if an image has >= 5 of them | Each hex nut has exactly one round hole. Measured: real holes > 0.9, the curved-triangle gaps between three touching nuts < 0.7 |
| 4. Nut piles | Every blob pixel goes to its nearest hole (hexagon-like cells); extra seed if a blob is a whole nut bigger than its holes explain | Splits piles of 350+ touching nuts cleanly |
| 5. Screws / bolts | Per-image area of one loose object (iterated trimmed median); blob count k = round(area/unit - 0.15); k distance-map peaks as seeds; marker watershed on **distance map + image brightness** | Brightness term makes cuts follow the bright rims where two parts actually overlap, instead of cutting straight across a screw |

## Results

Accuracy per image = `1 - |predicted - true| / true`. Ground truth is in `../ground_truth.csv`
(see "Ground truth" below).

| Image group | Images scored | Accuracy |
|---|---|---|
| `Screws_2024_07_15` (43 screws each) | 7 / 7 | 100 % on every image |
| Sparse screws (10-21 objects) | 4 / 4 | 100 % on every image |
| Nut piles (354 nuts) | 2 / 5 | 100 %, 100 % (all five give 354) |
| Mixed screws + upright screws | 3 / 6 | 99.0 %, 98.1 %, 99.0 % |
| Dense screws only (~196) | 1 / 3 | 98.5 % |
| **All scored images** | **17 / 25** | **mean 99.7 %, worst 98.1 %** |

The 8 images without ground truth are listed in `evaluate.py` output as not scored.

## Ground truth

The dataset has no labels, so I made them:

* **11 images counted by hand** from the raw photos (all `Screws_*` images and the four sparse
  ScrewAndBolt images). `img1_43_nosy`'s file name confirms 43.
* **6 dense images audited**: a dot is drawn on every detected object, and every grid cell is checked
  (zooming into clusters) for objects with no dot (miss), two dots (split) or one dot on two objects
  (merge). Ground truth = detections - extras + misses. Uncertain spots are noted in the CSV.
* Cross-check for nuts: a separate hole-only counter (round enclosed holes) also gives 353-354 on
  all five pile photos.

The audit found real bugs that were then fixed: gaps between touching nuts counted as holes
(fixed by the circularity test), gaps between touching screws treated as nut holes (fixed by the
">= 5 holes" rule), and one screw pair counted as one (fixed by relaxing the seed spacing).
It also corrected my own first count of one pile (355 -> 354).

## Known limitations

* **Overlapping screws hide area.** In the densest screw-only photos a pile of three overlapping
  screws can show the area of ~2.4 screws and be counted as 2 (about 1.5 % under-count).
* **Two object sizes in one image.** Upright screws (seen as discs) are bigger than lying screws,
  so a disc + screw blob is sometimes split into 3 (about 1-2 % over-count in the mixed photos).
* Rounding the area estimate differently fixes one case and breaks the other, so it is left
  neutral and documented instead of tuned per image.
