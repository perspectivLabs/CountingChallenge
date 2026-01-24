### **Overview of Computer Vision Counting**

This folder has some **computer vision stuff** for counting things like **screws and bolts** in pictures. It uses **OpenCV** but skips **deep learning models**, which is kind of nice if you want something simpler. I guess that's the main point here.

### **Traditional Processing: `solution_cv.py`**

The first file is **`solution_cv.py`**. It does **traditional image processing** to find and count objects.

* **Preprocessing**: It starts with turning the image to **grayscale** and adding **Gaussian blurring** to smooth it out.
* **Thresholding**: Then thresholding with **Otsu's method**, but inverse so the objects stand out.
* **Morphology**: After that, morphology operations—**opening** to remove noise and **dilation** to make the shapes more solid.
* **Contours**: Contours get detected next, just the **external ones**.
* **Filtering**: Finally, it **filters small contours by area** so you don’t count junk.

### **Usage for `solution_cv.py`**

For usage, the script looks at **directories** set in the main block. You have to change `dataset_screw_bolt` and `dataset_screws` to your own paths, I suppose. Run it with `python solution_cv.py`, and outputs go to `output_screw_bolt` and `output_screws` folders.

### **Advanced Segmentation: `solution_cv_watershed.py`**

Another one is **`solution_cv_watershed.py`**. This uses the **watershed algorithm**, which helps **separate touching objects** that might stick together in basic thresholding. It feels like a step up for messy images.

* **Preprocessing**: Preprocessing is the same (**grayscale and blurring**).
* **Thresholding**: Thresholding with **Otsu** again.
* **Distance Transform**: Then **distance transform** to figure out distances from zero pixels.
* **Markers**: Markers come from **sure foreground and background** areas, and **watershed splits the regions**.
* **Counting**: Counting happens by **unique markers** for each object.

### **Usage for `solution_cv_watershed.py`**

You run it on a **single image** like `python solution_cv_watershed.py` followed by the path. If you forget the path, it uses some **hardcoded one** inside.

### **Dependencies**

Dependencies are **Python 3.x**, **OpenCV** from `opencv-python`, and **NumPy**. Install with `pip install opencv-python numpy`. That seems straightforward, but I might be missing if there are version issues or something.

### **Summary**

Overall, these scripts show how to do counting **without fancy AI**, but the **watershed one handles overlaps better**. It gets a bit messy explaining the steps, though.
