# Non-AI Solution: Item Counting & Mask Overlay

**Candidate:** Athira PT  
**Task:** Count screws/bolts and overlay segmentation masks with > 95% accuracy using classical Computer Vision (OpenCV).

---

## 1. Overview & Challenge Analysis

In industrial vision, detecting and counting metallic hardware (screws, bolts, fasteners) on standard surfaces presents several computer vision challenges:
1. **Non-Uniform Illumination & Vignetting:** Typical office or industrial camera setups exhibit lighting falloff towards image boundaries and corners, causing simple global thresholding to fail.
2. **Specular Glare:** Polished steel reflects light directly into the camera lens, producing peak intensities (~255) inside dark objects that create hollow mask rings if unhandled.
3. **Complex Surface Texture & Threads:** Thread grooves create sharp, high-frequency intensity variations that cause edge detectors and adaptive thresholding to fragment a single screw into multiple pieces.
4. **Resolution Variance:** Images range from 1080p up to 12-megapixel phone photos (3000x4000).

---

## 2. Experiments Conducted & Method Selection Rationale

Four distinct classical CV pipelines were benchmarked in `non_ai_solution.ipynb`:

| Experiment | Technique | Lighting Invariance | Specular Hole Resilience | Over-segmentation Resistance | Accuracy on Screws | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Exp 1** | Global Otsu's Thresholding | Poor | Poor (hollow bodies) | Moderate | 78.4% | Rejected |
| **Exp 2** | Adaptive Gaussian Thresholding | Moderate | Poor | Poor (thread fragmentation) | 62.1% | Rejected |
| **Exp 3** | Distance Transform & Watershed | Moderate | Moderate | Very Poor (peaks on each thread) | 54.8% | Rejected |
| **Exp 4** | **Morphological Illumination Model + Differential Thresholding + FloodFill** | **Excellent** | **Excellent (FloodFill)** | **Excellent** | **99.0%** | **SELECTED** |

### Why Experiment 4 Was Selected:
- **Background Normalization:** By using a morphological structuring element larger than the screw widths, we extract the low-frequency background illumination surface without affecting object geometry. Subtraction yields a zero-mean background regardless of lighting gradients.
- **Solid Mask Guarantee:** The inverted flood-fill repair seamlessly closes internal specular reflection holes, preserving full object silhouettes.
- **Scale Invariance:** Resolution-aware kernel sizing ensures identical behavior across both 1080p and 4K images.

---

## 3. Pipeline Architecture

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
Specular Highlight Repair (Morphological Close + Inverted FloodFill)
      │
      ▼
Noise Removal (Morphological Open 3x3)
      │
      ▼
Contour Extraction & Scale-Adjusted Area Filtering
      │
      ▼
Multi-Color Translucent Mask Overlay (Alpha = 0.45) + Centroids + Bounding Boxes + Banner
```

---

## 4. Benchmark Results

### Screws Dataset (Ground Truth = 43 items per image):
| Filename | Detected Count | Ground Truth | Accuracy |
| :--- | :---: | :---: | :---: |
| `img1.jpg` | 43 | 43 | 100.0% |
| `img1_43_nosy.jpg` | 43 | 43 | 100.0% |
| `img2.jpg` | 43 | 43 | 100.0% |
| `img3.jpg` | 42 | 43 | 97.7% |
| `img4.jpg` | 42 | 43 | 97.7% |
| `img5.jpg` | 42 | 43 | 97.7% |
| `img6.jpg` | 43 | 43 | 100.0% |
| **Average Accuracy** | | | **99.0%** |

Target threshold of **> 95%** is successfully exceeded.

---

## 5. Directory Structure

```
Athira_PT/Non_AI/
├── count_and_mask.py          # Standalone Python CLI and module
├── non_ai_solution.ipynb      # Complete executed notebook with EDA & experiments
├── requirements.txt           # Non-AI Python dependencies
├── README.md                  # This documentation
└── output/                    # Annotated visual overlays and binary masks
    ├── Screws/
    └── ScrewAndBolt/
```

---

## 6. How to Run

### Installation:
```bash
pip install -r requirements.txt
```

### Run CLI on a Dataset:
```bash
# Process Screws dataset
python count_and_mask.py --input ../../data/Screws --output_dir output/Screws

# Process ScrewAndBolt dataset
python count_and_mask.py --input ../../data/ScrewAndBolt --output_dir output/ScrewAndBolt
```

### Run CLI on a Single Image:
```bash
python count_and_mask.py --input ../../data/Screws/img1.jpg --output_dir output/
```

### Run the Notebook:
```bash
jupyter notebook non_ai_solution.ipynb
```
