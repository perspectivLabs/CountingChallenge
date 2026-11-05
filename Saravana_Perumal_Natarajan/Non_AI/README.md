# Non-AI solution: classical image processing (OpenCV)

Counts the screws, bolts and nuts in each image and draws one mask per object, using only
thresholding, morphology, connected components, distance transforms and k-means. No machine learning.

## Run

Python 3.12, managed with [uv](https://docs.astral.sh/uv/). `requirements.txt` is exported from `uv.lock` for pip users.

```bash
cd Saravana_Perumal_Natarajan/Non_AI
uv sync
uv run count.py /path/to/data -o outputs      # overlays + counts.json
uv run evaluate.py /path/to/data -o outputs   # scores against ../ground_truth.json
```

`/path/to/data` is the unzipped dataset, which holds `ScrewAndBolt_20240713/` and `Screws_2024_07_15/`.
`non_ai_counting.ipynb` walks through the steps with pictures.

## Method

1. **Load.** Apply the EXIF orientation and resize so the long side is 2000 px.
2. **Foreground.**
   - Fit a degree-4 polynomial surface to the background and subtract it, to remove uneven lighting.
     The fit is redone twice so objects do not bias it.
   - Threshold the result with Otsu.
   - Add strongly saturated pixels, for brass screws.
   - Remove dark lens-vignette corners.
3. **One object's area.** The most common connected-component area (the peak of the log-area histogram).
4. **Image type.** It is a nut image if most isolated objects contain exactly one hole.
5. **Nuts.**
   - Find holes with a closing slightly larger than a hole. This works even when glare breaks the ring.
   - Give every pixel to the nearest hole within one nut radius (Voronoi split).
   - A square gap between four nuts can look like a hole, but its cell gets only slivers of the
     neighbouring nuts. So the weakest cell under 0.7 of a nut is dropped and the cells are recomputed,
     one at a time.
   - Solid leftovers of at least half a nut count as nuts whose hole is hidden.
6. **Screws and bolts.**
   - Round objects (bolt heads, upright screws) are distance-transform peaks close to their radius;
     a lying screw is too thin to make one.
   - The rest of a touching group counts as `floor(area / screw area + 0.35)` screws. A further screw needs
     0.65 of a screw's area, because a group's outline and shadow add area.
   - The group is split into that many masks with k-means.
7. **Overlay.** Draw each object in its own colour, with the count in the corner.

## Results

Scored against `../ground_truth.json` (see `../GroundTruth/README.md`).

| Image set | Images | True / predicted | Mean accuracy | Worst image |
|---|---|---|---|---|
| Brass screws (`Screws_2024_07_15`) | 7 | 301 / 301 | 100.00% | 100.00% |
| Sparse screws (`192951`, `193135`, `193650`, `193659`) | 4 | 62 / 62 | 100.00% | 100.00% |
| Dense screw piles (`193831`, `193839`, `193907`) | 3 | 583 / 592 | 98.45% | 97.93% |
| Screws + bolt heads (`194200` to `194316`) | 6 | 591 / 594 | 99.15% | 97.96% |
| Hex nuts (`194541` to `194630`) | 5 | 1770 / 1769 | 99.94% | 99.72% |
| **All** | **25** | **3307 / 3318** | **99.60%** | 97.93% |

Every image is above 95%. The weakest are the dense screw piles, where touching screws are sometimes
split into one piece too many.

[`results/`](results/) holds sample overlays and the per-image scores (`results.json`).

## Limitations

- Area-based splitting assumes the objects in one image are about the same size.
- Masks inside a touching group are straight k-means cuts, so their outlines are approximate.
- A nut lying completely on another one, with its hole hidden, is counted by area.
- The two thresholds (0.7 of a nut, 0.65 of a screw) were checked against the same 25 images reported here.
  There is no separate test set.
