# Submission checklist

- [ ] Forked `perspectivLabs/CountingChallenge` (not verifiable in this workspace)
- [x] Project folder named `Sathish_Nampelli_CountingChallenge`
- [x] AI FastSAM-S solution, requirements, README, and notebook present
- [x] Non-AI OpenCV solution, requirements, README, and notebook present
- [x] Root README documents methods, commands, outputs, metrics, and limitations
- [x] Full dataset is outside the project; two local sample photos are under the
  ignored `Non_AI/single_test/` path. Dataset, virtual environments, model
  checkpoints, caches, and temporary annotation progress are ignored
- [x] Both pipelines completed all 25 images; prediction CSVs have 25 successful rows
- [x] Mask/overlay outputs exist for all 25 images for both methods
- [x] Prediction filenames and output dimensions validated against the original images
- [x] AI and Non-AI evaluation reports regenerated using the same verified labels
- [x] Eight verified labels are in `ground_truth.csv`; four provisional labels are
  recorded separately in `ground_truth_plan.csv`
- [x] Local point-annotation tool saves progress and keeps provisional labels out of
  formal evaluation
- [ ] Remaining 17 images have independently verified ground truth
- [ ] Assignment's >95% target demonstrated (current verified-set results are below
  95%; full-dataset accuracy is not established)
- [ ] Pull request raised from the fork (not verifiable in this workspace)

## Current verified-set results

| Method | Verified / 25 | Mean relative accuracy | MAE | MAPE | Exact matches |
|---|---:|---:|---:|---:|---:|
| AI FastSAM-S | 8 | 89.26% | 1.750 | 10.74% | 3 / 8 |
| Non-AI OpenCV | 8 | 85.39% | 2.375 | 14.61% | 2 / 8 |

These are partial results only. The 17 unverified images and provisional counts are
not included in formal accuracy.
