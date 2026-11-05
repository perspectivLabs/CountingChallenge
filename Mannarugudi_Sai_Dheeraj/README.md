# Hardware Counting Assessment

I developed two approaches to count screws and other hardware in images:

- **Non-AI:** OpenCV thresholding and area-based counting with watershed splitting.
- **AI:** YOLO instance segmentation.

## Run the OpenCV counter

From the repository root, install the requirements and run the counter:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r Mannarugudi_Sai_Dheeraj\Non_AI\requirements.txt
.\.venv\Scripts\python.exe Mannarugudi_Sai_Dheeraj\Non_AI\count_items.py --input Mannarugudi_Sai_Dheeraj\data --output Mannarugudi_Sai_Dheeraj\Non_AI\output
```

The script saves count results, masks, and overlays in `Non_AI\output`.

## Train and run the AI model

YOLO needs instance-segmentation annotations. The generated OpenCV labels are only provisional and should be manually reviewed and corrected before training.

```powershell
py -3.12 -m venv .venv-ai
.\.venv-ai\Scripts\python.exe -m pip install -r Mannarugudi_Sai_Dheeraj\AI\requirements.txt
.\.venv-ai\Scripts\python.exe Mannarugudi_Sai_Dheeraj\AI\generate_pseudo_labels.py
```

After correcting the labels and checking `AI\hardware.yaml`, train the model:

```powershell
.\.venv-ai\Scripts\python.exe Mannarugudi_Sai_Dheeraj\AI\train.py --data Mannarugudi_Sai_Dheeraj\AI\hardware.yaml --epochs 100 --imgsz 1024
```

Use the `best.pt` file from the run folder created by training:

```powershell
.\.venv-ai\Scripts\python.exe Mannarugudi_Sai_Dheeraj\AI\predict.py --weights path\to\best.pt --input Mannarugudi_Sai_Dheeraj\data
```

Predictions and overlays are saved in `AI\predictions`.

## Accuracy

I have not verified accuracy above 95%. To measure exact-count accuracy, manually count a separate held-out test set, create a CSV with `image,true_count,predicted_count` columns, and run:

```powershell
.\.venv\Scripts\python.exe Mannarugudi_Sai_Dheeraj\evaluate_counts.py --csv path\to\held_out_counts.csv
```