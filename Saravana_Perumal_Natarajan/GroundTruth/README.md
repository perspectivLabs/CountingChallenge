# Ground truth

The dataset comes without labels. `../ground_truth.json` holds one point per object for all 25 images.
It was built by cross-checking two independent counters and checking only their disagreements by eye.
Nobody hand-counted hundreds of nuts.

1. **Two counters.**
   - The Non-AI counter (`../Non_AI/count.py`).
   - `sam_count.py`: SAM 2 (`sam2.1_b`) prompted with points on dark pixels, on overlapping 512 px tiles.
     SAM 2 is a general segmenter, so its mistakes are unrelated to those of the Non-AI counter.
2. **Match** (`compare.py prepare`). Pair instances one-to-one by mask IoU (Hungarian matching, IoU >= 0.3).
   Pairs are agreed objects. Overlapping unmatched instances are merged into one candidate, and each
   candidate is drawn on a zoomed review tile (`work/cand/`).
3. **Review.** For each tile, check whether the outline holds a real object without an agreed dot.
   The accepted candidates, with their points, are in `review.json`.
4. **Second pass on the medium-confidence images.** Ground-truth points were compared with all three
   counters (Non-AI, SAM 2, the AI model) and every disagreement was looked at again.
   - Added 9 missed objects, 6 of them in `193907`.
   - Removed 2 double points, in `194232`.
   Both are stored in `review.json` (`added`, `removed`).
5. **Build** (`compare.py finalize`). Ground truth = agreed + accepted + added − removed.

```bash
uv sync
uv run sam_count.py /path/to/data         # about 30 s per image on Apple MPS; downloads sam2.1_b.pt
uv run compare.py prepare /path/to/data   # agreed objects, candidates, review tiles
uv run compare.py finalize                # writes ../ground_truth.json
```

`ground_truth.json` is the frozen reference that both solutions are scored against. Candidate ids
depend on the Non-AI version used at review time, so re-running `prepare` after changing the
Non-AI counter gives new candidates to review.

## Summary

| Image group | Agreed by both | Candidates reviewed | Ground truth | Confidence |
|---|---|---|---|---|
| Brass screws (7) | 43 each | 6, all rejected (SAM 2 on vignette corners) | 43 | high |
| Sparse screws (4) | all | 0 | 21, 21, 10, 10 | high |
| Hex nuts (5) | 351–354 | 75: 8 nuts SAM 2 missed, the rest gaps between nuts | 354 in every photo | high |
| Dense screw piles (3) | 184–185 | 59 | 193, 196, 194 | medium |
| Screws + bolt heads (6) | 90–101 | 74 | 98, 100, 98, 99, 98, 98 | medium |

- **Nuts:** the five photos show the same nuts, and each one gives exactly 354. That is a strong
  check on the nut decisions.
- **Screw piles:** deciding whether a disputed outline is a separate screw is sometimes a judgement call.
- **Cross-check with other submissions.** Other candidates published hand counts in their pull requests
  to the challenge repository (PRs #20, #22 and #23).
  - Brass, sparse and nut images: identical to #20 and #23. #22 differs by 1 to 2 on two nut images.
  - Medium-confidence images, after the second pass: within ±1 of #20 and #23, and within ±2 of #22.
  - Their counts were used only to find where to look again. Every change was decided by looking at the image.
- **The AI solution:** it is trained on labels from the same agreement, so its scores measure agreement
  with this reference.
