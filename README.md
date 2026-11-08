# Counting Challenge

This project counts and segments screws, bolts, and nuts in images using an AI method and a classical computer-vision method. The original task sets a target accuracy above 95%; no accuracy report or benchmark is included here, so results should be checked against labeled images.

## Repository layout

- `Srihari_M/AI/`: SAM3 text-prompt segmentation and an annotation viewer. See [AI README](Srihari_M/AI/README.md).
- `Srihari_M/Non_AI/`: OpenCV/SciPy-based image processing and counting. See [Non-AI README](Srihari_M/Non_AI/README.md).
- `Srihari_M/data/`: local sample images for the Non-AI script. See [sample-data README](Srihari_M/data/README.md).
- `Srihari_M/`: project overview and links to both approaches. See [project README](Srihari_M/README.md).

The AI dataset and model checkpoint are excluded from Git. Download the dataset separately and provide the SAM3 checkpoint before running the AI command. The checkpoint link placeholder is in [SAM3_CHECKPOINT.txt](Srihari_M/AI/SAM3_CHECKPOINT.txt). The original challenge dataset is [available on Google Drive](https://drive.google.com/drive/folders/1TuM4CgGI3WBqOHNFjuzjaEzLOH5Yw43_?usp=sharing).

## Quick start

Run the classical pipeline from the repository root:

```bash
python3 -m pip install -r Srihari_M/Non_AI/requirements.txt
python3 Srihari_M/Non_AI/Non_AI_Counting.py --input Srihari_M/data --output Srihari_M/Non_AI/Output/run_01 --upto 8
```

The final overlay images and `counts.csv` will be under `Srihari_M/Non_AI/Output/run_01/`. See the folder README for what each stage shows.

For AI setup, install a PyTorch/Ultralytics environment that supports SAM3, place the checkpoint at `Srihari_M/AI/sam3.pt`, and run the command documented in [AI README](Srihari_M/AI/README.md). AI dependencies and weights are not installed or downloaded by this repository.

## Data and generated files

Input images, dataset files, model weights, and generated outputs are intentionally not committed. The source dataset for the YOLO annotations is identified in `Srihari_M/AI/dataset/README.dataset.txt`; the challenge dataset is also linked there in the original project notes. Use the README in each project folder for exact paths, commands, and expected results.
