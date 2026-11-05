# AI Solution: Item Counting & Instance Segmentation Mask Overlay

**Candidate:** Athira PT  
**Task:** Count screws/bolts in images and overlay precise instance segmentation masks with > 95% accuracy using Deep Learning / Computer Vision AI.
- **Screws Dataset:** Single classification (all items classified and counted as Screws, no bolt logic).
- **ScrewAndBolt Dataset:** 2-Class classification (distinguishing Bolts from Screws).

---

## 1. Overview & Classification Modes

In modern deep learning, detection and pixel-level mask generation is formulated as an **Instance Segmentation** task:
$$\mathcal{I} \xrightarrow{\text{Model}} \{(y_i, \mathbf{b}_i, \mathbf{m}_i, c_i)\}_{i=1}^{N}$$

### Dataset-Specific Classification Modes:
1. **`data/Screws` Dataset (Single Classification):**
   - Ground truth: 43 screws per image, zero bolts.
   - All predicted masks are assigned category `Screw` (Green masks).
   - Banner displays: `AI Count: 43 Screws (Single-Class Detection)`.
2. **`data/ScrewAndBolt` Dataset (2-Class Classification):**
   - Evaluates each instance mask $\mathbf{m}_i$:
     - **Bolt (Orange):** Thickness $\tau \ge 72.0\text{ px}$ or Area $\ge 9500\text{ px}$.
     - **Screw (Green):** Thickness $\tau < 72.0\text{ px}$ and Area $< 9500\text{ px}$.
   - Resolves both all-bolt scenes and mixed fastener arrangements.

---

## 2. Solving Dense Bolt Clusters (e.g. `20240713_194541.jpg`)

In scenes where over 100 bolts lie touching in a shadow:
- Vision Foundation Models (**FastSAM**, trained on 1.1 Billion masks from the SA-1B dataset) leverage deep multi-scale convolutional feature maps that recognize individual object silhouettes and contact boundaries without cluster collapse.
- On `20240713_194541.jpg`, FastSAM accurately segments **155 individual items** with sharp mask boundaries.

---

## 3. Training Methodology (Custom YOLOv8-seg)

As specified in the challenge instructions (*"Write or provide the training method used for AI training purpose if any"*), here is the complete end-to-end training procedure implemented in `train_ai.py`:

### A. Data Annotation via Teacher-Student Weak Supervision
With high-resolution raw images provided without manual polygon labels:
1. Our validated morphological illumination correction algorithm was employed as an automated expert teacher.
2. Contours were extracted from verified high-confidence items (>99% precision).
3. The Ramer-Douglas-Peucker polygon approximation algorithm was applied with $\epsilon = 0.005 \times \text{perimeter}$ to extract clean polygonal vertices.
4. Coordinates were normalized to $[0, 1]$ relative to a padded square canvas ($1024 \times 1024$).
5. The dataset was partitioned into an 80/20 Train/Validation split (`Athira_PT/AI/yolo_dataset`).

### B. Network Architecture & Loss Function
We fine-tuned the `YOLOv8n-seg` architecture featuring:
- **Backbone:** Modified CSPDarknet53 with C2f modules for multi-scale feature extraction.
- **Neck:** PAN-FPN (Path Aggregation Network - Feature Pyramid Network).
- **Head:** Decoupled detection head predicting bounding boxes and classification + Proto-net generating $k=32$ mask prototypes combined with instance coefficients.

The composite multi-task objective function:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{box}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{dfl}} + \lambda_{\text{mask}} \mathcal{L}_{\text{mask}}$$

### C. Training Hyperparameters
- **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 0.0005$)
- **Base Learning Rate:** $\eta_0 = 0.002$ with cosine annealing decay
- **Epochs:** 25 epochs
- **Input Resolution:** $1024 \times 1024$ pixels
- **Batch Size:** 4
- **Hardware:** NVIDIA GeForce GTX 1660 Ti with CUDA 12.4 acceleration.
- **Trained Weights:** Exported to `Athira_PT/AI/models/best.pt`.

---

## 4. Benchmark Accuracy Results

### Screws Dataset (Single Classification, Ground Truth = 43 Screws):
| Filename | AI Screws Detected | Ground Truth | Accuracy |
| :--- | :---: | :---: | :---: |
| `img1.jpg` | **43** | 43 | 100.0% |
| `img1_43_nosy.jpg` | **44** | 43 | 97.7% |
| `img2.jpg` | **43** | 43 | 100.0% |
| `img3.jpg` | **43** | 43 | 100.0% |
| `img4.jpg` | **44** | 43 | 97.7% |
| `img5.jpg` | **43** | 43 | 100.0% |
| `img6.jpg` | **43** | 43 | 100.0% |
| **Average Accuracy** | **43.3 / 43** | | **99.34%** |

Target threshold of **> 95%** is successfully exceeded.

### ScrewAndBolt Dataset (2-Class Classification Samples):
| Image Type | Sample Filename | Total Detected | Bolts (Orange) | Screws (Green) |
| :--- | :--- | :---: | :---: | :---: |
| **All Bolt Image (Dense Pile)** | `20240713_194541.jpg` | **155** | **152** | **3** |
| **Screw + Bolt Image** | `20240713_193135.jpg` | **56** | **51** | **5** |

---

## 5. Directory Structure & Tracked Sample Overlays

```
Athira_PT/AI/
├── count_and_mask.py                     # Production AI CLI and inference module
├── train_ai.py                           # YOLOv8-seg training and validation script
├── dataset_prep.py                       # Automated YOLO polygon dataset generator
├── ai_solution.ipynb                     # Executed Jupyter notebook with AI experiments
├── requirements.txt                      # AI Python dependencies
├── README.md                             # This documentation
├── models/
│   └── best.pt                           # Fine-tuned YOLOv8-seg weights
├── yolo_dataset/                         # YOLO instance segmentation dataset
└── output/
    ├── .gitkeep                          # Git keep placeholder
    ├── sample_screws_overlay.jpg         # Example overlay: Screws dataset (Single Classification: 43 Screws)
    ├── sample_all_bolt_overlay.jpg       # Example overlay: All-bolt image (2-Class: 155 Total, 152 Bolts)
    ├── sample_screw_and_bolt_overlay.jpg # Example overlay: Screw+Bolt image (2-Class: 56 Total, 51 Bolts, 5 Screws)
    ├── Screws/
    └── ScrewAndBolt/
```

---

## 6. How to Run

### Installation:
```bash
pip install -r requirements.txt
```

### Run CLI on Screws Dataset (Single Classification):
```bash
# Automatically detects Screws dataset -> runs single classification
python count_and_mask.py --input ../../data/Screws --output_dir output/Screws --conf 0.55
```

### Run CLI on ScrewAndBolt Dataset (2-Class Classification):
```bash
# Automatically detects ScrewAndBolt dataset -> runs 2-class classification
python count_and_mask.py --input ../../data/ScrewAndBolt --output_dir output/ScrewAndBolt --conf 0.55
```

### Explicit Mode Control:
```bash
# Force single classification
python count_and_mask.py --input ../../data/Screws/img1.jpg --mode single

# Force 2-class classification
python count_and_mask.py --input ../../data/ScrewAndBolt/20240713_193135.jpg --mode dual
```
