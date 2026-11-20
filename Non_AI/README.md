# Non AI solutions

I used Opencv to apply and test multiple filters to help me separate foreground from background. Since all 25 images have dark metallic nuts and screws on a plain light background and each image has either only screws or only nuts, classical computer vision on CPU felt like a practical approach worth exploring before jumping to heavy models.

NOTE: Everything here runs completely on CPU using just OpenCV and NumPy, no GPU needed.

## Grayscale and watershed separation
I did try using the grayscale and watershed separation first (`watershed.py`), but it showed unreliable masking on screws where it created separate regions in some screws for the head and the main body. Because screws are elongated, a plain distance transform ends up giving multiple peaks along a single screw and splits it up. Uneven lighting and vignetting near the corners also made simple global thresholding pretty flaky.

## Basic geometric filtering
Next I tried estimating the background with morphological closing and subtracting it out (`geometric.py`), then using contour area, circularity and inner holes to count nuts and screws. This handled the uneven lighting much better, but once parts started touching tightly or overlapping in dense piles it still struggled to separate them cleanly.
## Adaptive geometric segmentation
To handle touching nuts and overlapping screws properly, I put together an adaptive pipeline inside the `adaptive_pipeline` folder and ran it through `run_adaptive_segmentation.py`.

### Illumination normalization and foreground mask
First step is dealing with shadows, vignetting and uneven lighting. I estimate a smooth background using morphological closing with border replication and gaussian blur, then divide the grayscale image by that background instead of just subtracting. After that I combine Otsu and adaptive thresholding with light morphological opening and closing so small dust gets removed without filling in the center holes of the nuts.

### Classifying isolated vs touching objects
From the foreground mask I extract connected components and compute basic geometric properties like area, aspect ratio, circularity, solidity, contour hierarchy for internal holes, distance transform and Zhang-Suen skeleton. Based on these properties each component gets routed into one of three cases: isolated objects, touching nuts, or touching and overlapping screws. Isolated screws and nuts are taken directly as individual masks.

### Separating touching nuts
For touching nuts, relying on raw distance peaks does not work great because of the hollow centers. Instead I use the internal hole of each nut as the instance marker. Candidate holes from contour hierarchy are validated by size, circularity, surrounding foreground ring and minimum spacing between centers (so random gaps trapped between three touching nuts do not get counted as nuts). Then marker controlled watershed splits the touching nuts along their contact seams, and intersecting back with the original mask keeps the internal holes intact.

### Separating touching and overlapping screws
Since simple distance transform over-segments screws, I tested five different approaches on touching screw clusters and saved the outputs for each separately so I could compare them: skeleton graph analysis, Hough line detection, rotation invariant template matching, contour concavity analysis and geometrically constrained watershed.

For the final adaptive output, I first cut touching screws along sharp contour concavity necks, and for whatever pieces still stay merged (like crossed or parallel touching screws) I refine them using template matching, constrained watershed and skeleton graph, picking whichever matches the expected single screw area best.

### Results
The final adaptive pipeline runs completely on CPU (under a second for the 1080p images and around 3 to 6 seconds for the 3000x4000 images) and gave solid counts across the dataset. It got exact 43 on all the brass screw images even with vignetting and noise, 10 on the sparse black screws, around 98-101 and 199-201 on the dense screw piles, and 355-370 on the dense nut trays.

If the production setup keeps the same dark items on light background conditions, this is easily the cheapest and cleanest option since it needs no training and no GPU.

Results with all intermediate stages, screw separation comparisons and final overlays are here: https://drive.google.com/drive/folders/1VviYM4Z9FsMNW3g7qE70NTTgIa-WjEPx?usp=sharing
