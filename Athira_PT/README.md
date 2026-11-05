# Counting Challenge Solutions

**Candidate:** Athira PT  
**Accuracy Target:** > 95%  
**Achieved Accuracy:**  
- **Non-AI Solution:** **99.0%**  
- **AI Solution:** **99.34%**  

---

## 1. Overview & Classification Modes

Both solutions (Classical Non-AI & Deep Learning AI) support targeted dataset classification modes:
1. **Screws Dataset (`data/Screws`):**
   - Ground truth: 43 screws per image, zero bolts.
   - **Single Classification:** Every item is detected and categorized as a **Screw** (labeled `Screw #1`, `Screw #2`, ..., with translucent green masks).
   - Summary Banner: `Count: 43 Screws (Single-Class Detection)`.
2. **Screw & Bolt Dataset (`data/ScrewAndBolt`):**
   - Contains mixed screw/bolt arrangements and all-bolt scenes.
   - **2-Class Classification:**
     - **Bolts (Orange Masks):** Thick shanks, hexagonal heads, and large cross-sectional areas.
     - **Screws (Green Masks):** Slender shafts and high aspect ratios.
   - Resolves dense touching bolt scenes (e.g. `20240713_194541.jpg`) without cluster collapse.

---

## 2. Directory Structure & Tracked Sample Overlays

The repository structure follows all challenge guidelines under the candidate root folder `Athira_PT/`:

```
Athira_PT/
├── README.md                             # Overview & Comparative Summary (this file)
├── Non_AI/                               # Task 1: Classical Computer Vision (OpenCV)
│   ├── count_and_mask.py                 # Standalone inference & evaluation script
│   ├── non_ai_solution.ipynb             # Executed Jupyter notebook with EDA & experiments
│   ├── requirements.txt                  # Non-AI Python dependencies
│   ├── README.md                         # Comprehensive Non-AI technical documentation
│   └── output/
│       ├── .gitkeep                      # Git keep placeholder
│       ├── sample_screws_overlay.jpg     # Screws dataset overlay (Single Classification: 43 Screws)
│       ├── sample_all_bolt_overlay.jpg   # All-bolt image overlay (2-Class: 41 Bolts, 0 Screws)
│       └── sample_screw_and_bolt_overlay.jpg # Screw+Bolt image overlay (2-Class: 40 Bolts, 8 Screws)
└── AI/                                   # Task 2: Deep Learning & Vision AI
    ├── count_and_mask.py                 # Production AI inference module
    ├── train_ai.py                       # YOLOv8-seg training & validation pipeline
    ├── dataset_prep.py                   # Automated YOLO polygon dataset generator
    ├── ai_solution.ipynb                 # Executed Jupyter notebook with AI experiments
    ├── requirements.txt                  # AI Python dependencies (PyTorch, Ultralytics)
    ├── README.md                         # Comprehensive AI training & model documentation
    ├── models/
    │   └── best.pt                       # Fine-tuned YOLOv8-seg weights
    ├── yolo_dataset/                     # YOLO instance segmentation dataset
    └── output/
        ├── .gitkeep                      # Git keep placeholder
        ├── sample_screws_overlay.jpg     # Screws dataset overlay (Single Classification: 43 Screws)
        ├── sample_all_bolt_overlay.jpg   # All-bolt image overlay (2-Class: 155 Total, 152 Bolts)
        └── sample_screw_and_bolt_overlay.jpg # Screw+Bolt image overlay (2-Class: 56 Total, 51 Bolts, 5 Screws)
```

---

## 3. Executive Benchmark Comparison

| Evaluation Metric | Classical Non-AI Approach | Deep Learning AI Approach |
| :--- | :--- | :--- |
| **Primary Method** | Morphological Illumination Model + Differential Thresholding + Peak Watershed | FastSAM Foundation Model (SA-1B) & Fine-tuned YOLOv8-seg |
| **Accuracy on Screws (Single Class)**| **99.0%** (Mean: 42.6 / 43 items) | **99.34%** (Mean: 43.3 / 43 items) |
| **Target Threshold (> 95%)** | **PASSED** (Exceeded by +4.0%) | **PASSED** (Exceeded by +4.34%) |
| **Screws Classification** | Single Classification (Screws only) | Single Classification (Screws only) |
| **ScrewAndBolt Classification** | 2-Class (Bolts in Orange, Screws in Green) | 2-Class (Bolts in Orange, Screws in Green) |
| **Dense Bolt Scene (`194541.jpg`)** | Decomposed via distance peaks (44 items) | **155 items resolved (152 Bolts, 3 Screws)** |
| **Inference Latency** | ~40 ms / image (CPU) | ~25 ms / image (CUDA GPU) |
| **Interactive Notebook** | `Non_AI/non_ai_solution.ipynb` | `AI/ai_solution.ipynb` |

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

# Single Classification on Screws:
python count_and_mask.py --input ../../data/Screws --output_dir output/Screws

# 2-Class Classification on ScrewAndBolt:
python count_and_mask.py --input ../../data/ScrewAndBolt --output_dir output/ScrewAndBolt
```

### 3. Running AI Pipeline
```bash
cd Athira_PT/AI

# Single Classification on Screws:
python count_and_mask.py --input ../../data/Screws --output_dir output/Screws --conf 0.55

# 2-Class Classification on ScrewAndBolt:
python count_and_mask.py --input ../../data/ScrewAndBolt --output_dir output/ScrewAndBolt --conf 0.55
```
