# AI Solution: Object Counting & Instance Segmentation Mask Overlay

**Candidate:** Athira PT  
**Task:** Count objects (screws, bolts, fasteners) and overlay precise instance segmentation masks with > 95% accuracy using a Pretrained AI Vision Foundation Model (FastSAM).

---

## 1. Overview & AI Formulation

The task requires detecting every hardware object in the image, producing a pixel-level segmentation mask, and reporting the exact count $N$. In modern deep learning, this is structured as an **Instance Segmentation** task:
$$\mathcal{I} \xrightarrow{\text{Model}} \{(\mathbf{b}_i, \mathbf{m}_i, c_i)\}_{i=1}^{N}$$

Where each detected object $i$ is characterized by:
- Bounding box $\mathbf{b}_i = [x_1, y_1, x_2, y_2]$
- Binary segmentation mask $\mathbf{m}_i \in \{0, 1\}^{H \times W}$
- Confidence probability $c_i \in [0, 1]$
- Item label index $\#i$

---

## 2. Why a Pretrained Foundation Model (FastSAM)

Fine-tuning deep neural networks from scratch on very small datasets (<25 images) is well-documented in machine learning literature to introduce significant drawbacks:
1. **Severe Overfitting:** Deep backbones memorize the few training backgrounds rather than learning general fastener semantics.
2. **Pseudo-Label Noise:** Teacher-generated polygon annotations introduce synthetic edge artifacts that degrade mask boundary fidelity.
3. **Fragility to Lighting Shifts:** Fine-tuned small models break under variations in flash angle and shadows.

Instead, we utilize **FastSAM (Fast Segment Anything Model)**:
- Pretrained on the massive **SA-1B dataset (11 Million images, 1.1 Billion masks)**.
- Encodes deep spatial representations that generalize zero-shot to metallic hardware.
- Robust against specular glare on shiny steel surfaces.
- Naturally resolves touching and overlapping objects in dense scenes (e.g. `20240713_194541.jpg`).
- Achieves **99.34% counting accuracy** on the ground-truth benchmark without requiring fragile fine-tuning.

---

## 3. Benchmark Accuracy Results

### Screws Dataset (Ground Truth = 43 Objects per Image):
| Filename | AI Detected Count | Ground Truth | Accuracy |
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

---

## 4. Directory Structure & Tracked Overlays

```
Athira_PT/AI/
├── count_and_mask.py                     # Standalone Python CLI and inference module
├── ai_solution.ipynb                     # Pre-rendered executed notebook (2.79 MB)
├── requirements.txt                      # AI Python dependencies
├── README.md                             # This documentation
└── output/
    ├── .gitkeep                          # Git keep placeholder
    ├── sample_screws_overlay.jpg         # Example overlay: Screws dataset (43 Objects)
    ├── sample_screw_and_bolt_overlay.jpg # Example overlay: ScrewAndBolt dataset (56 Objects)
    ├── Screws/
    └── ScrewAndBolt/
```

---

## 5. How to Run

### Installation:
```bash
pip install -r requirements.txt
```

### Run CLI on Screws Dataset:
```bash
python count_and_mask.py --input ../../data/Screws --output_dir output/Screws --conf 0.55
```

### Run CLI on ScrewAndBolt Dataset:
```bash
python count_and_mask.py --input ../../data/ScrewAndBolt --output_dir output/ScrewAndBolt --conf 0.55
```

### Run CLI on a Single Image:
```bash
python count_and_mask.py --input ../../data/Screws/img1.jpg --output_dir output/
```

### Run the Notebook:
```bash
jupyter notebook ai_solution.ipynb
```
