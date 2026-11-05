# CountingChallenge: Saravana Perumal Natarajan

Count the screws, bolts and nuts in each image and draw a mask on every one, in two ways:

| Folder | Approach | Details |
|---|---|---|
| [`Non_AI/`](Non_AI/) | Classical image processing (OpenCV): background fit, Otsu, hole-based nut counting, area-based screw counting | [README](Non_AI/README.md) |
| [`AI/`](AI/) | YOLO11n-seg fine-tuned on 640 px tiles, with automatically generated labels and tiled inference | [README](AI/README.md) |
| [`GroundTruth/`](GroundTruth/) | How `ground_truth.json` was built: Non-AI and SAM 2 cross-check, then a visual review | [README](GroundTruth/README.md) |

Each folder is its own [uv](https://docs.astral.sh/uv/) project (Python 3.12) with `pyproject.toml`, `uv.lock` and an exported
`requirements.txt`. The code is linted and formatted with ruff (`ruff.toml`).

## Quick start

```bash
(cd Non_AI && uv sync && uv run count.py /path/to/data -o outputs)   # Non-AI
(cd AI && uv sync && uv run count.py /path/to/data -o outputs)       # AI (uses model/items_yolo11n_seg.pt)
```

`/path/to/data` is the unzipped dataset (`ScrewAndBolt_20240713/`, `Screws_2024_07_15/`). The data is not committed.
Each `count.py` writes one overlay per image and a `counts.json` file. Each `evaluate.py` also scores the counts
against `ground_truth.json`.

## Results

Count accuracy per image = 1 − |predicted − true| / true.

| Solution | Mean accuracy (25 images) | Worst image | Total count (true 3307) |
|---|---|---|---|
| Non-AI | 99.60% | 97.93% (`20240713_193831`) | 3318 |
| AI | 99.53% | 95.24% (`20240713_193135`) | 3299 |
| AI, 5 held-out test images | 99.90% | 99.48% | 698 of 699 |

### Per-image counts

Diff = predicted − true.

| Image | Group | AI split | Ground truth | GT confidence | Non-AI | Diff | Non-AI acc | AI | Diff | AI acc |
|---|---|---|---|---|---|---|---|---|---|---|
| `img1.jpg` | Brass screws | train | **43** | high | 43 | 0 | 100.00% | 43 | 0 | 100.00% |
| `img1_43_nosy.jpg` | Brass screws | train | **43** | high | 43 | 0 | 100.00% | 43 | 0 | 100.00% |
| `img2.jpg` | Brass screws | **test** | **43** | high | 43 | 0 | 100.00% | 43 | 0 | 100.00% |
| `img3.jpg` | Brass screws | train | **43** | high | 43 | 0 | 100.00% | 43 | 0 | 100.00% |
| `img4.jpg` | Brass screws | train | **43** | high | 43 | 0 | 100.00% | 43 | 0 | 100.00% |
| `img5.jpg` | Brass screws | val | **43** | high | 43 | 0 | 100.00% | 43 | 0 | 100.00% |
| `img6.jpg` | Brass screws | train | **43** | high | 43 | 0 | 100.00% | 43 | 0 | 100.00% |
| `20240713_192951.jpg` | Sparse screws | train | **21** | high | 21 | 0 | 100.00% | 21 | 0 | 100.00% |
| `20240713_193135.jpg` | Sparse screws | train | **21** | high | 21 | 0 | 100.00% | 22 | +1 | 95.24% |
| `20240713_193650.jpg` | Sparse screws | val | **10** | high | 10 | 0 | 100.00% | 10 | 0 | 100.00% |
| `20240713_193659.jpg` | Sparse screws | **test** | **10** | high | 10 | 0 | 100.00% | 10 | 0 | 100.00% |
| `20240713_193831.jpg` | Dense screw pile | train | **193** | medium | 197 | +4 | 97.93% | 193 | 0 | 100.00% |
| `20240713_193839.jpg` | Dense screw pile | val | **196** | medium | 197 | +1 | 99.49% | 193 | −3 | 98.47% |
| `20240713_193907.jpg` | Dense screw pile | **test** | **194** | medium | 198 | +4 | 97.94% | 193 | −1 | 99.48% |
| `20240713_194200.jpg` | Screws + bolt heads | train | **98** | medium | 100 | +2 | 97.96% | 97 | −1 | 98.98% |
| `20240713_194206.jpg` | Screws + bolt heads | train | **100** | medium | 100 | 0 | 100.00% | 97 | −3 | 97.00% |
| `20240713_194215.jpg` | Screws + bolt heads | train | **98** | medium | 98 | 0 | 100.00% | 98 | 0 | 100.00% |
| `20240713_194232.jpg` | Screws + bolt heads | train | **99** | medium | 100 | +1 | 98.99% | 98 | −1 | 98.99% |
| `20240713_194256.jpg` | Screws + bolt heads | val | **98** | medium | 99 | +1 | 98.98% | 98 | 0 | 100.00% |
| `20240713_194316.jpg` | Screws + bolt heads | **test** | **98** | medium | 97 | −1 | 98.98% | 98 | 0 | 100.00% |
| `20240713_194541.jpg` | Hex nuts | train | **354** | high | 354 | 0 | 100.00% | 354 | 0 | 100.00% |
| `20240713_194551.jpg` | Hex nuts | train | **354** | high | 354 | 0 | 100.00% | 354 | 0 | 100.00% |
| `20240713_194606.jpg` | Hex nuts | train | **354** | high | 353 | −1 | 99.72% | 354 | 0 | 100.00% |
| `20240713_194621.jpg` | Hex nuts | val | **354** | high | 354 | 0 | 100.00% | 354 | 0 | 100.00% |
| `20240713_194630.jpg` | Hex nuts | **test** | **354** | high | 354 | 0 | 100.00% | 354 | 0 | 100.00% |
| **All 25** | | | **3307** | | **3318** | +11 | **99.60%** | **3299** | −8 | **99.53%** |
| **Test images (5)** | | | **699** | | 702 | +3 | 99.38% | 698 | −1 | **99.90%** |

- **AI split:** the AI model never saw the `test` images in training. The `val` images were used to choose the model
  and its confidence threshold. The Non-AI method has no training, so the split does not apply to it.
- **GT confidence:**
  - `high`: an unambiguous count. The nuts and brass screws were photographed several times and always give the same count.
  - `medium`: tangled screw piles, where a few objects are a judgement call.
  - See [GroundTruth/README.md](GroundTruth/README.md), including a cross-check against counts published by other candidates.
- **More detail:** mask-level checks (matches, merges, misses, false positives) are in `Non_AI/results/results.json`
  and `AI/results/results.json`. Sample overlays are in the same folders.
