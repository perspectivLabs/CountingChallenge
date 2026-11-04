# Non-AI solution (classical OpenCV)

`count_objects.py` counts the objects in every image and writes an overlay with one coloured mask per object.

```bash
pip install -r requirements.txt
python count_objects.py --data ../../data --out outputs
# -> outputs/<image>.jpg (masks + count) and outputs/counts.csv
```

It runs on the CPU, about 0.4 s per image. It has no learned parameters.

## Method

1. **Resize.** The long side is set to 1600 px, so the kernel sizes stay the same for every photo.
2. **Flatten the illumination.** The method divides the grayscale image by a background estimate (a morphological closing, then a large blur). This removes uneven lighting and lens vignetting.
3. **Find the foreground.** An Otsu threshold separates the objects, which are darker than the background. A small opening and closing removes noise, and holes are filled. The black vignette corners are masked out.
4. **Nuts (one round hole = one nut).** Holes are the inner contours of the foreground. A hole counts if it is round (circularity > 0.75) and of a consistent size; the irregular gaps between touching parts fail this test. A blob with *n* round holes holds *n* nuts. It is split into masks by nearest hole centre (a Voronoi split).
5. **Screws and bolts (blobs without holes).**
   - The area of one object is the smallest well-populated mode of the log-area histogram. This handles scenes where most blobs are touching pairs.
   - **Discs** are modelled separately. Isolated round blobs give the disc area and radius. Inside a cluster, each distance-transform core thicker than 0.8 × the disc radius is one disc; screw shanks and heads are much thinner.
   - The remaining area is divided by the single-screw area. Clusters (2.5 or more objects) get a 6 % crowding allowance, because shadows and edges between touching objects add area. The measured ratio is 1.05 to 1.07 per object on clusters of 3 or more.
   - A cluster is split into *k* masks with a distance-transform watershed.

## Results

Accuracy per image is `1 - |pred - gt| / gt`. The ground truth is in `../ground_truth.csv` (see `../AI/README.md` for how it was made).

| set | images | mean accuracy | worst image |
|---|---|---|---|
| all | 25 | **99.7 %** | 98.0 % |
| test split (held out for the AI model) | 5 | 99.8 % | 99.0 % |

Run `python ../evaluate.py outputs/counts.csv` for the per-image table.

Known limits:

- Watershed boundaries in large clusters are approximate; the count is right, but the mask edges may not follow the true object edges.
- The per-image object size is inferred, so a scene of only touching pairs with no single objects would fail.
