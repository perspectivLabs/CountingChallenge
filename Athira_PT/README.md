# Counting Challenge Solutions

**Candidate:** Athira PT  
**Accuracy Target:** > 95%  
**Achieved Accuracy:**  
- **Non-AI Solution (OpenCV):** **99.0%**  
- **AI Solution (FastSAM Foundation Model):** **99.34%**  

---

## 1. Problem Statement & Scope

The objective of this challenge is to:
1. Detect and count all hardware objects in the input images.
2. Generate and overlay precise pixel-level segmentation masks for each detected item.
3. Deliver two complete solutions:
   - **Non-AI:** Classical Computer Vision using OpenCV (illumination modeling + differential thresholding + peak watershed).
   - **AI:** Pretrained Vision Foundation Model (FastSAM trained on 1.1 Billion masks) for zero-shot instance segmentation without small-data fine-tuning.
4. Exceed the **> 95% accuracy** requirement on ground-truth benchmark data.

---

## 2. Directory Structure & Tracked Deliverables

All code, notebooks, requirements, and visual artifacts are organized under the candidate root folder `Athira_PT/`:

```
Athira_PT/
├── README.md                             # Comparative Summary & Quick Start (this file)
├── Non_AI/                               # Task 1: Classical Computer Vision (OpenCV)
│   ├── count_and_mask.py                 # Standalone inference & evaluation script
│   ├── non_ai_solution.ipynb             # Pre-rendered executed Jupyter notebook (3.26 MB)
│   ├── requirements.txt                  # Non-AI Python dependencies
│   ├── README.md                         # Non-AI technical documentation
│   └── output/
│       ├── .gitkeep                      # Git keep placeholder
│       ├── sample_screws_overlay.jpg     # Example overlay: Screws dataset (43 Objects)
│       └── sample_screw_and_bolt_overlay.jpg # Example overlay: ScrewAndBolt dataset (48 Objects)
└── AI/                                   # Task 2: Deep Learning & Vision AI
    ├── count_and_mask.py                 # Production AI inference module
    ├── ai_solution.ipynb                 # Pre-rendered executed Jupyter notebook (2.79 MB)
    ├── requirements.txt                  # AI Python dependencies
    ├── README.md                         # AI technical documentation
    └── output/
        ├── .gitkeep                      # Git keep placeholder
        ├── sample_screws_overlay.jpg     # Example overlay: Screws dataset (43 Objects)
        └── sample_screw_and_bolt_overlay.jpg # Example overlay: ScrewAndBolt dataset (56 Objects)
```

---

## 3. Executive Benchmark Comparison

| Evaluation Metric | Classical Non-AI Approach | Deep Learning AI Approach |
| :--- | :--- | :--- |
| **Primary Method** | Morphological Illumination Model + Differential Thresholding + Peak Watershed | Pretrained FastSAM (SA-1B Zero-Shot Instance Segmentation) |
| **Accuracy on Benchmark (`data/Screws`)** | **99.0%** (Mean: 42.6 / 43 items) | **99.34%** (Mean: 43.3 / 43 items) |
| **Target Threshold (> 95%)** | **PASSED** (Exceeded by +4.0%) | **PASSED** (Exceeded by +4.34%) |
| **Training Requirements** | None (rule-based CV) | None (pretrained foundation model, no fine-tuning needed) |
| **Dense Cluster Resolution** | Distance Transform peak decomposition | Deep multi-scale feature maps |
| **Inference Latency** | ~40 ms / image (CPU) | ~25 ms / image (CUDA GPU) |
| **Executed Jupyter Notebook** | `Non_AI/non_ai_solution.ipynb` | `AI/ai_solution.ipynb` |

---

## 4. Quick Start Guide

### 1. Environment Setup
```bash
git clone https://github.com/a-pt/CountingChallenge.git
cd CountingChallenge

python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux/macOS

pip install -r requirements.txt
```

### 2. Running Non-AI Pipeline
```bash
cd Athira_PT/Non_AI

# Run on Screws dataset:
python count_and_mask.py --input ../../data/Screws --output_dir output/Screws

# Run on ScrewAndBolt dataset:
python count_and_mask.py --input ../../data/ScrewAndBolt --output_dir output/ScrewAndBolt

# Run on a single image:
python count_and_mask.py --input ../../data/Screws/img1.jpg --output_dir output/
```

### 3. Running AI Pipeline
```bash
cd Athira_PT/AI

# Run on Screws dataset:
python count_and_mask.py --input ../../data/Screws --output_dir output/Screws --conf 0.55

# Run on ScrewAndBolt dataset:
python count_and_mask.py --input ../../data/ScrewAndBolt --output_dir output/ScrewAndBolt --conf 0.55

# Run on a single image:
python count_and_mask.py --input ../../data/Screws/img1.jpg --output_dir output/
```
