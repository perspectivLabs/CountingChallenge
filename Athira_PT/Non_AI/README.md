# Non-AI Solution: Item Counting & Mask Overlay (OpenCV)

**Candidate:** Athira PT  
**Task:** Count screws/bolts and overlay segmentation masks with > 95% accuracy using classical Computer Vision (OpenCV).
- **Screws Dataset:** Single classification (all items classified and counted as Screws, no bolt logic).
- **ScrewAndBolt Dataset:** 2-Class classification (distinguishing Bolts from Screws).

---

## 1. Overview & Classification Modes

In industrial vision, detecting metallic fasteners presents several challenges:
1. **Specular Glare:** Polished metal reflects light directly into the lens, causing hollow mask rings if unhandled.
2. **Dense Touching Clusters (e.g. `20240713_194541.jpg`):** Tightly packed bolts in shadows can form enclosed perimeters that merge into a single component if global floodfill is applied.
3. **Dataset-Specific Categorization:**
   - **`data/Screws` Dataset:** All items are screws (ground truth = 43 screws per image, zero bolts). The pipeline runs **single classification** (labeling every item as `Screw #i` with green masks and reporting total screws).
   - **`data/ScrewAndBolt` Dataset:** Images contain both bolts and screws or all-bolt arrangements. The pipeline runs **2-class classification**:
     - **Bolt (Orange):** Wide diameter ($\tau \ge 52.0\text{ px}$), large area ($\ge 2800\text{ px}$), or compact aspect ratio ($< 3.2$).
     - **Screw (Green):** Slender shaft (aspect ratio $\ge 3.2$, $\tau < 52.0\text{ px}$, area $< 2800\text{ px}$).

---

## 2. Experiments Conducted & Method Selection Rationale

Four distinct classical CV pipelines were benchmarked in `non_ai_solution.ipynb`:

| Experiment | Technique | Lighting Invariance | Specular Hole Resilience | Over-segmentation Resistance | Accuracy on Screws | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Exp 1** | Global Otsu's Thresholding | Poor | Poor (hollow bodies) | Moderate | 78.4% | Rejected |
| **Exp 2** | Adaptive Gaussian Thresholding | Moderate | Poor | Poor (thread fragmentation) | 62.1% | Rejected |
| **Exp 3** | Distance Transform & Watershed | Moderate | Moderate | Very Poor (peaks on each thread) | 54.8% | Rejected |
| **Exp 4** | **Morphological Illumination Model + Differential Thresholding + Peak Watershed** | **Excellent** | **Excellent** | **Excellent** | **99.0%** | **SELECTED** |

---

## 3. Resolving the Dense Bolt Cluster Undercounting

### Root Cause of the "#16 Cluster Collapse":
In earlier naive approaches, a global border floodfill (`cv2.floodFill(diff_mask, seed=(0, 0))`) was applied to fill specular highlights.
- In `20240713_194541.jpg`, a large cluster of touching bolts was cast in shadow.
- The dense perimeter of touching bolts formed an enclosed barrier preventing the global floodfill from reaching interior background gaps.
- Consequently, the floodfill inverted the entire central interior as one huge "hole" and filled it solid (~322,000 pixels).
- The 15 isolated peripheral bolts were detected as items 1–15, and the entire central mass was labeled as item **#16**.

### Engineering Solution:
1. **Contour-Level Specular Filling:** We extract external contours and fill only genuine internal holes per contour using `cv2.drawContours(..., thickness=cv2.FILLED)`. This completely eliminates cross-item bridging.
2. **Distance-Transform Peak Decomposition:** For oversized clusters ($\text{Area} > 2.2 \times \text{median}$), we compute the Euclidean Distance Transform:
   $$\mathcal{D}(x, y) = \min_{(x', y') \in \text{Background}} \sqrt{(x - x')^2 + (y - y')^2}$$
   We detect topological local maxima ($h$-maxima peak filtering) to identify distinct physical fastener cores, then partition touching bolts via marker-controlled watershed.

---

## 4. Pipeline Architecture

```
Input Image (RGB)
      │
      ▼
Scale Normalization (target dimension 1600 px)
      │
      ▼
Grayscale & Gaussian Smoothing (5x5 kernel)
      │
      ▼
Morphological Illumination Modeling (Large Elliptical Dilation)
      │
      ▼
Differential Subtraction: diff = bg_model - blurred
      │
      ▼
Differential Thresholding (diff > threshold)
      │
      ▼
Contour Extraction & Specular Highlight Hole Filling (cv2.drawContours FILLED)
      │
      ▼
Cluster Detection & Distance Transform Peak Decomposition (for dense clusters)
      │
      ▼
Classification:
  - If mode == 'single' (Screws dataset): All items -> Screw (Green)
  - If mode == 'dual'   (ScrewAndBolt):   Geometry test -> Bolt (Orange) vs Screw (Green)
      │
      ▼
Color-Coded Translucent Mask Overlay + Centroids + Summary Banner
```

---

## 5. Benchmark Results

### Screws Dataset (Single Classification, Ground Truth = 43 Screws):
| Filename | Screws Detected | Ground Truth | Accuracy |
| :--- | :---: | :---: | :---: |
| `img1.jpg` | **43** | 43 | 100.0% |
| `img1_43_nosy.jpg` | **43** | 43 | 100.0% |
| `img2.jpg` | **43** | 43 | 100.0% |
| `img3.jpg` | **42** | 43 | 97.7% |
| `img4.jpg` | **42** | 43 | 97.7% |
| `img5.jpg` | **42** | 43 | 97.7% |
| `img6.jpg` | **43** | 43 | 100.0% |
| **Average Accuracy** | **42.6 / 43** | | **99.0%** |

Target threshold of **> 95%** is successfully exceeded.

### ScrewAndBolt Dataset (2-Class Classification Samples):
| Image Type | Sample Filename | Total Detected | Bolts (Orange) | Screws (Green) |
| :--- | :--- | :---: | :---: | :---: |
| **All Bolt Image** | `20240713_194551.jpg` | **41** | **41** | **0** |
| **All Bolt Scene (Dense)** | `20240713_194541.jpg` | **44** | **42** | **2** |
| **Screw + Bolt Image** | `20240713_193135.jpg` | **48** | **40** | **8** |
| **Screw + Bolt Image** | `20240713_192951.jpg` | **33** | **30** | **3** |

---

## 6. Directory Structure & Tracked Sample Overlays

```
Athira_PT/Non_AI/
├── count_and_mask.py                     # Standalone Python CLI and module
├── non_ai_solution.ipynb                 # Complete executed notebook with EDA & experiments
├── requirements.txt                      # Non-AI Python dependencies
├── README.md                             # This documentation
└── output/
    ├── .gitkeep                          # Git keep placeholder
    ├── sample_screws_overlay.jpg         # Example overlay: Screws dataset (Single Classification: 43 Screws)
    ├── sample_all_bolt_overlay.jpg       # Example overlay: All-bolt image (2-Class: 41 Bolts, 0 Screws)
    ├── sample_screw_and_bolt_overlay.jpg # Example overlay: Screw+Bolt image (2-Class: 40 Bolts, 8 Screws)
    ├── Screws/
    └── ScrewAndBolt/
```

---

## 7. How to Run

### Installation:
```bash
pip install -r requirements.txt
```

### Run CLI on Screws Dataset (Single Classification):
```bash
# Automatically detects Screws dataset -> runs single classification
python count_and_mask.py --input ../../data/Screws --output_dir output/Screws
```

### Run CLI on ScrewAndBolt Dataset (2-Class Classification):
```bash
# Automatically detects ScrewAndBolt dataset -> runs 2-class classification
python count_and_mask.py --input ../../data/ScrewAndBolt --output_dir output/ScrewAndBolt
```

### Explicit Mode Control:
```bash
# Force single classification
python count_and_mask.py --input ../../data/Screws/img1.jpg --mode single

# Force 2-class classification
python count_and_mask.py --input ../../data/ScrewAndBolt/20240713_193135.jpg --mode dual
```
