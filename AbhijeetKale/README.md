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
The true count of 43 was confirmed by a manual count on img3 and by the file name img1_43_nosy.jpg.
Full counts and overlays: `Non_AI/results/` and `AI/results/`.

## Limitations
- Dense piles in ScrewAndBolt (hundreds of touching screws and nuts) are approximate for both methods.
  Hidden objects cannot be counted from a top-down 2D image. SAM undercounts touching screws there,
  and the area-ratio estimate in Non_AI is approximate.
- Separated objects (Screws_2024_07_15 and the sparse ScrewAndBolt images) count reliably.