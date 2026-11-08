# Non-AI Counting

## What it does

`Non_AI_Counting.py` counts a single object type per image using image processing, not a trained model. It estimates the background from the image border, thresholds object pixels, cleans the mask, finds connected regions, splits touching objects, filters fragments, and classifies the image as nut or screw/bolt by shape. It saves intermediate images so each processing step can be checked.

## Setup and run

Install the packages listed in `requirements.txt` and run from the repository root:

```bash
python3 -m pip install -r Srihari_M/Non_AI/requirements.txt
python3 Srihari_M/Non_AI/Non_AI_Counting.py --input Srihari_M/data --output Srihari_M/Non_AI/Output/run_01 --upto 8
```

For one image, replace the input folder with an image path, for example:

```bash
python3 Srihari_M/Non_AI/Non_AI_Counting.py --input Srihari_M/data/img1.jpg --output Srihari_M/Non_AI/Output/one_image --upto 8
```

`--input` accepts one image or a folder of images. `--output` chooses where results are written. `--upto` runs only through the specified stage from 1 to 8; it defaults to 8. Use `--upto 3` to inspect preprocessing, background distance, and thresholding without completing object counting.

## What to inspect

Each stage writes results into its own output subfolder:

| Folder | Result |
| --- | --- |
| `01_preprocess/` | Denoised image with edges preserved. |
| `02_distance_map/` | Colorized distance from the estimated background. |
| `03_threshold/` | Initial foreground/background mask. |
| `04_clean_mask/` | Mask after noise removal and cleanup. |
| `05_components/` | Separate connected regions, with IDs; a CSV of region measurements is also saved here. |
| `06_split/` | Regions after attempting to split touching objects. |
| `07_filter_classify/` | Filtered regions and the image-level class. |
| `08_overlay/` | Final colored mask overlay with predicted class and count. |
| `08_label_map/` | Lossless label image; each nonzero pixel value is an object ID. |

At stage 8, `counts.csv` records the image, final count, predicted class, and split statistics. Check the overlays and label maps for missed objects, merged objects, or background regions incorrectly counted. Outputs are generated locally and ignored by Git.

The sample images in `Srihari_M/data/` are not included in Git. For other image sizes or backgrounds, the thresholds and shape assumptions in the script may need tuning; the stated accuracy target is not guaranteed by the pipeline.