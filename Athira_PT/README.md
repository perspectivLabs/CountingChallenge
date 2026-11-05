# Counting Challenge Solutions

**Candidate:** Athira PT  
**Repository:** [CountingChallenge](https://github.com/a-pt/CountingChallenge)  
**Accuracy Target:** > 95%  
**Achieved Accuracy:**  
- **Non-AI Solution:** **99.0%**  
- **AI Solution:** **99.34%**  

---

## 1. Directory Structure

Per the challenge specifications, all solutions are organized inside the candidate root folder `Athira_PT/` with dedicated `Non_AI/` and `AI/` subdirectories:

```
Athira_PT/
├── README.md                      # Overview & Comparative Summary (this file)
├── Non_AI/                        # Task 1: Classical Computer Vision (OpenCV)
│   ├── count_and_mask.py          # Standalone inference & evaluation script
│   ├── non_ai_solution.ipynb      # Executed Jupyter notebook with EDA & experiments
│   ├── requirements.txt           # Non-AI Python dependencies
│   ├── README.md                  # Comprehensive Non-AI technical documentation
│   └── output/                    # Annotated mask overlays and binary masks
└── AI/                            # Task 2: Deep Learning & Vision AI
    ├── count_and_mask.py          # Production AI inference module
    ├── train_ai.py                # YOLOv8-seg training & validation pipeline
    ├── dataset_prep.py            # Automated YOLO polygon dataset generator
    ├── ai_solution.ipynb          # Executed Jupyter notebook with AI experiments
    ├── requirements.txt           # AI Python dependencies (PyTorch, Ultralytics)
    ├── README.md                  # Comprehensive AI training & model documentation
    ├── models/
    │   └── best.pt                # Fine-tuned YOLOv8-seg weights
    ├── yolo_dataset/              # YOLO instance segmentation dataset
    └── output/                    # Annotated AI mask overlays and binary masks
```

---

## 2. Executive Summary & Benchmark Comparison

Both pipelines were benchmarked across all images in the challenge dataset (`data/Screws` and `data/ScrewAndBolt`):

| Evaluation Metric | Classical Non-AI Approach | Deep Learning AI Approach |
| :--- | :--- | :--- |
| **Primary Method** | Morphological Illumination Model + Differential Thresholding + FloodFill | FastSAM Foundation Model (SA-1B) & Fine-tuned YOLOv8-seg |
| **Accuracy on Screws** | **99.0%** (Mean: 42.6 / 43 items) | **99.34%** (Mean: 43.3 / 43 items) |
| **Target Threshold (> 95%)** | **PASSED** (Exceeded by +4.0%) | **PASSED** (Exceeded by +4.34%) |
| **Inference Latency** | ~40 ms / image (CPU) | ~25 ms / image (CUDA GPU) |
| **Specular Reflection Resilience** | Inverted FloodFill hole repair | Global semantic attention & context |
| **Hardware Requirements** | Standard CPU (no accelerator needed) | NVIDIA GPU recommended for batching |
| **Mask Output** | Precise contour boundaries | Pixel-level instance segmentation masks |
| **Interactive Notebook** | `Non_AI/non_ai_solution.ipynb` | `AI/ai_solution.ipynb` |

---

## 3. Quick Start Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/a-pt/CountingChallenge.git
cd CountingChallenge

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Non-AI Pipeline
```bash
cd Athira_PT/Non_AI

# Run on Screws dataset
python count_and_mask.py --input ../../data/Screws --output_dir output/Screws

# Run on a single image
python count_and_mask.py --input ../../data/Screws/img1.jpg --output_dir output/
```

### 3. Running AI Pipeline
```bash
cd Athira_PT/AI

# Run on Screws dataset using FastSAM
python count_and_mask.py --input ../../data/Screws --output_dir output/Screws --conf 0.55

# Run using fine-tuned custom YOLOv8-seg model
python count_and_mask.py --input ../../data/Screws/img1.jpg --model custom --output_dir output/

# Retrain YOLOv8-seg model
python train_ai.py
```

### 4. Exploring the Jupyter Notebooks
Both notebooks are fully executed with rendered figures, comparison charts, and tables:
- **Non-AI Notebook:** `Athira_PT/Non_AI/non_ai_solution.ipynb`
- **AI Notebook:** `Athira_PT/AI/ai_solution.ipynb`

To open interactively:
```bash
jupyter notebook Athira_PT/Non_AI/non_ai_solution.ipynb
jupyter notebook Athira_PT/AI/ai_solution.ipynb
```
