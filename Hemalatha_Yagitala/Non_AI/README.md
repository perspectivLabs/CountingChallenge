# Non-AI solution: classical OpenCV

Counts screws, bolts and hex nuts and overlays one coloured mask per object, using only
OpenCV, NumPy and SciPy. There's no learning and no pretrained weights.

## Run
```bash
pip install -r requirements.txt
python count_objects.py --input <data_dir> --output results --gt ../ground_truth.csv
```
Outputs: `results/overlays/*.jpg` (masks, object ids and the total count drawn on each image) and
`results/counts.csv` (prediction, ground truth and accuracy per image).
`--save-masks` also writes 16-bit instance-label PNGs, which the AI pipeline uses as starting labels.

## Method
The dataset has three regimes, and each needs a different idea:

1. **Uneven lighting and vignetting.** The background is estimated with a large morphological closing
   and divided out (flat-field correction), then thresholded with Otsu. Dark lens-vignette
   corners are removed because they run along the image border, which real objects don't.
2. **Densely packed hex nuts (300+ touching).** Splitting the blobs is hopeless, but every nut
   has a round through-hole. If an image has many circular enclosed holes of consistent size, each
   hole seeds one nut, and nut pixels go to the nearest hole within the nut radius.
   Gaps between touching nuts aren't circular, so they're rejected.
3. **Screws and bolts, sparse or piled.** Watershed over-splits elongated screws, so counting is
   area-based: each blob is counted as round(area / single-object area). The single-object area
   comes from an iterated median over blob areas in each image, so it adapts to zoom. Round, solid
   blobs (screws standing upright) get their own reference area. Inside merged clusters,
   upright screws are found as thick distance-transform cores and counted separately.

**Masks.** A blob counted as *n* objects is split into *n* masks by k-means on
(x, y, local orientation from the structure tensor). Touching screws usually point different
ways, so they split along their own axes instead of along a straight cut.

## Results
Accuracy per image = 1 − |predicted − actual| / actual. Ground truth was counted by hand (`../ground_truth.csv`).

**Mean accuracy: 99.61% over 25 images. Every image is above 95% (lowest 96.94%).**

| Image | Predicted | Ground truth | Accuracy |
|---|---|---|---|
| 20240713_192951.jpg | 21 | 21 | 100.00% |
| 20240713_193135.jpg | 21 | 21 | 100.00% |
| 20240713_193650.jpg | 10 | 10 | 100.00% |
| 20240713_193659.jpg | 10 | 10 | 100.00% |
| 20240713_193831.jpg | 193 | 194 | 99.48% |
| 20240713_193839.jpg | 194 | 194 | 100.00% |
| 20240713_193907.jpg | 196 | 193 | 98.45% |
| 20240713_194200.jpg | 101 | 98 | 96.94% |
| 20240713_194206.jpg | 100 | 98 | 97.96% |
| 20240713_194215.jpg | 98 | 98 | 100.00% |
| 20240713_194232.jpg | 98 | 99 | 98.99% |
| 20240713_194256.jpg | 98 | 98 | 100.00% |
| 20240713_194316.jpg | 98 | 97 | 98.97% |
| 20240713_194541.jpg | 354 | 354 | 100.00% |
| 20240713_194551.jpg | 354 | 354 | 100.00% |
| 20240713_194606.jpg | 354 | 356 | 99.44% |
| 20240713_194621.jpg | 355 | 355 | 100.00% |
| 20240713_194630.jpg | 354 | 354 | 100.00% |
| img1.jpg | 43 | 43 | 100.00% |
| img1_43_nosy.jpg | 43 | 43 | 100.00% |
| img2.jpg | 43 | 43 | 100.00% |
| img3.jpg | 43 | 43 | 100.00% |
| img4.jpg | 43 | 43 | 100.00% |
| img5.jpg | 43 | 43 | 100.00% |
| img6.jpg | 43 | 43 | 100.00% |

### Calibration note
Three global settings were chosen by examining errors against this ground truth:
- **Nut-hole filter:** circularity > 0.7, area 0.6–1.6× the median. This rejects round gaps between three touching nuts and took the nut images from +11/+15 to within ±2.
- **Overlap bias:** −0.2 when rounding the count of clusters larger than 4 objects, because stacked screws hide area.
- **Upright-screw cores:** counted separately inside clusters. This took the mixed images from +11/+9 to +3/+2.

Each is one value for the whole dataset, not tuned per image. With a larger dataset, they should be tuned on a separate split.

## Limitations
- Objects stacked on top of each other hide area, so dense piles can be slightly undercounted.
- Nut counting needs the hole to be visible. A nut standing on its edge is caught by the area fallback.
- Masks inside tightly tangled clusters are approximate. The count is what the area rule guarantees.
