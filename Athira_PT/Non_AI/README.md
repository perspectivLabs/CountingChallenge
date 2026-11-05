# Non-AI Solution: Object Counting & Mask Overlay (OpenCV)

**Candidate:** Athira PT  
**Task:** Count objects (screws, bolts, fasteners) and overlay precise segmentation masks with > 95% accuracy using classical Computer Vision (OpenCV).

---

## 1. Overview & Problem Formulation

In industrial computer vision, automated item counting and quality inspection often encounter challenging imaging conditions:
1. **Non-Uniform Illumination & Vignetting:** Spatial lighting falloff causes simple global thresholding to miss edge items or missegment backgrounds.
2. **Specular Glare:** Polished steel reflects direct illumination, producing peak intensities (~255) inside dark objects that create hollow mask rings if unhandled.
3. **Dense Touching Fasteners:** When objects touch or overlap, naive connected components merge them into a single blob.
4. **Shadows & Cluster Bridging:** Low-angle lighting casts shadows across touching objects that bridge gaps under naive global floodfilling.

---

## 2. Experiments Conducted & Method Selection Rationale

Four distinct classical CV pipelines were benchmarked in `non_ai_solution.ipynb`:

| Experiment | Technique | Lighting Invariance | Specular Hole Resilience | Over-segmentation Resistance | Accuracy on Benchmark | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Exp 1** | Global Otsu's Thresholding | Poor | Poor (hollow bodies) | Moderate | 78.4% | Rejected |
| **Exp 2** | Adaptive Gaussian Thresholding | Moderate | Poor | Poor (thread fragmentation) | 62.1% | Rejected |
| **Exp 3** | Distance Transform & Watershed | Moderate | Moderate | Very Poor (peaks on each thread) | 54.8% | Rejected |
| **Exp 4** | **Morphological Illumination Model + Differential Thresholding + Peak Watershed** | **Excellent** | **Excellent** | **Excellent** | **99.0%** | **SELECTED** |

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
Contour Extraction & Specular Highlight Hole Filling (cv2.drawContours FILLED)
      │
      ▼
Cluster Detection & Distance Transform Peak Watershed (for touching clusters)
      │
      ▼
Translucent Mask Overlay + Centroids + Bounding Boxes + Object Count Banner
```

---

## 4. Benchmark Accuracy Results

### Screws Dataset (Ground Truth = 43 Objects per Image):
| Filename | Detected Count | Ground Truth | Accuracy |
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

---

## 5. Directory Structure & Tracked Overlays

```
Athira_PT/Non_AI/
├── count_and_mask.py                     # Standalone Python CLI and module
├── non_ai_solution.ipynb                 # Pre-rendered executed notebook (3.26 MB)
├── requirements.txt                      # Non-AI Python dependencies
├── README.md                             # This documentation
└── output/
    ├── .gitkeep                          # Git keep placeholder
    ├── sample_screws_overlay.jpg         # Example overlay: Screws dataset (43 Objects)
    ├── sample_screw_and_bolt_overlay.jpg # Example overlay: ScrewAndBolt dataset (48 Objects)
    ├── Screws/
    └── ScrewAndBolt/
```

---

## 6. How to Run

### Installation:
```bash
pip install -r requirements.txt
```

### Run CLI on Screws Dataset:
```bash
python count_and_mask.py --input ../../data/Screws --output_dir output/Screws
```

### Run CLI on ScrewAndBolt Dataset:
```bash
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
