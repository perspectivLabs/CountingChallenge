# Hardware Object Counting

PerspectivLabs CountingChallenge solution for counting screws, bolts, nuts, and
washers in 25 photographs. It provides two pipelines: pretrained FastSAM instance
segmentation and a traditional OpenCV segmentation/watershed approach. The full
dataset is outside this project. Two local sample photos remain under the ignored
`Non_AI/single_test/` path.

## Current evaluation

Both methods were run on the same 25 original images. The formal evaluation includes
only the eight rows in `ground_truth.csv`; four additional labels are provisional in
`ground_truth_plan.csv` and are excluded.

| Method | Verified / total | Unverified | Mean relative accuracy | MAE | MAPE | Exact matches |
|---|---:|---:|---:|---:|---:|---:|
| FastSAM AI | 8 / 25 | 17 | 89.26% | 1.750 | 10.74% | 3 / 8 (37.50%) |
| OpenCV Non-AI | 8 / 25 | 17 | 85.39% | 2.375 | 14.61% | 2 / 8 (25.00%) |

Neither result demonstrates the assignment's **>95%** target. The remaining 17
images have no verified labels. Earlier partial scores are superseded by these
evaluations; provisional counts were not scored. The existing verified
`20240713_193831.jpg = 102` label was retained in addition to the seven newly supplied
verified labels.

## Project contents

- `AI/ai_counting.py` — pretrained FastSAM-S inference, filtering, duplicate-mask
  handling, conservative large-mask splitting, and visual output.
- `Non_AI/non_ai_counting.py` — OpenCV candidate segmentation, morphology,
  connected components, watershed splitting, and visual output.
- `annotate_ground_truth.py` — local point-annotation tool displaying only original
  images; saves progress and separates provisional from verified labels.
- `evaluate_counts.py` — shared per-image evaluation and summary metrics.
- `ground_truth.csv` — verified labels only; the evaluator reads this file.
- `ground_truth_plan.csv` — verified and provisional label status.
- `AI/README.md` and `Non_AI/README.md` — method-specific setup and details.
- `validation_contact_sheet.jpg` — representative original images and output overlays.

## Install

Use Python 3.10 or later. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r AI\requirements.txt
```

`AI/requirements.txt` includes the OpenCV dependencies and FastSAM's PyTorch and
Ultralytics packages. The Non-AI pipeline and annotator can instead use
`Non_AI/requirements.txt`. The annotation GUI uses Tkinter and Pillow.

## Run both pipelines

Point `--input` at the extracted dataset root containing the two image folders:

```powershell
python AI\ai_counting.py --input "<DATASET_ROOT>" --output AI\final_output
python Non_AI\non_ai_counting.py --input "<DATASET_ROOT>" --output Non_AI\final_output
```

FastSAM downloads `FastSAM-s.pt` on first use if it is not already available.
Each output folder contains `predictions.csv`, `evaluation.csv`, and one
`_mask.png` and `_overlay.jpg` for every input image. Outputs are grouped by source
image folder; the full input path is recorded in each predictions CSV.

Run evaluation after both predictions are generated:

```powershell
python evaluate_counts.py `
  --predictions AI\final_output\predictions.csv `
  --ground-truth ground_truth.csv `
  --output AI\final_output\evaluation.csv

python evaluate_counts.py `
  --predictions Non_AI\final_output\predictions.csv `
  --ground-truth ground_truth.csv `
  --output Non_AI\final_output\evaluation.csv
```

## Ground-truth annotation

Launch the local tool against the dataset root:

```powershell
python annotate_ground_truth.py --input "<DATASET_ROOT>"
```

Left-click once at each object's center. The tool shows numbered marks and a running
count, supports undo/clear, previous/next navigation, mouse-wheel zoom, and middle- or
right-drag panning. Point progress is saved in ignored `annotation_progress.json`.
Choose **Save verified** only after independently checking the count; **Save
provisional** stores a working label without adding it to formal evaluation. The tool
never loads predictions, masks, or overlays.

## Metric definition

For each positive verified count, relative accuracy is
`1 - abs(prediction - ground_truth) / ground_truth`. Overall relative counting
accuracy is the mean of per-image relative accuracies multiplied by 100. MAPE is the
mean absolute error divided by ground truth, in percent. MAE is the mean absolute
count error. Exact-match percentage uses verified images as its denominator. The
reports include per-image errors and verification/prediction status.

## Limitations

FastSAM is class-agnostic and can merge touching parts or omit small objects.
Traditional thresholding and watershed can under-segment touching items or
over-segment textured/dense scenes. Small samples and piles of overlapping hardware
remain difficult. Metrics describe only the eight verified images and are not an
estimate of full-dataset performance.
