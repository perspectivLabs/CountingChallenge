# Srihari_M Project

This folder contains two ways to count hardware parts in images, plus sample images.

| Path | Purpose |
| --- | --- |
| `AI/` | SAM3 prompt-based segmentation and counting, plus an annotation viewer. See [AI README](AI/README.md). |
| `Non_AI/` | Classical image-processing pipeline with eight stages. See [Non-AI README](Non_AI/README.md). |
| `data/` | Local sample images used for the Non-AI example. See [data README](data/README.md). |

The AI dataset and checkpoint, Non-AI outputs, and sample input files are excluded from Git. Obtain the AI dataset and checkpoint separately before running AI inference. Commands below should be run from the repository root unless stated otherwise.

## Non-AI example

```bash
python3 -m pip install -r Srihari_M/Non_AI/requirements.txt
python3 Srihari_M/Non_AI/Non_AI_Counting.py --input Srihari_M/data --output Srihari_M/Non_AI/Output/run_01 --upto 8
```

Look in `Srihari_M/Non_AI/Output/run_01/08_overlay/` for the final counted images and in `counts.csv` for the per-image count and predicted class.

## AI example

After placing the SAM3 checkpoint at `Srihari_M/AI/sam3.pt` and obtaining the dataset, use the command in [AI README](AI/README.md). The checked-in first-line example refers to a `Sample_data` folder that is not part of this repository; the README gives a command using the dataset path instead.