
import cv2
import numpy as np
import sys


# added padding with background colour to prevent edge-clipping of objects
def _add_border(img: np.ndarray, size: int = 30) -> np.ndarray:
    
    bg = img[10, 10].tolist()
    return cv2.copyMakeBorder(img, size, size, size, size,
                              cv2.BORDER_CONSTANT, value=bg)


def _watershed_split(mask: np.ndarray, dist_thresh: float = 0.45) -> int:
    
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return 1
    _, sure_fg = cv2.threshold(dist, dist_thresh * dist.max(), 255, 0)
    sure_fg    = np.uint8(sure_fg)
    sure_bg    = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    unknown    = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers    = markers + 1
    markers[unknown == 255] = 0
    bgr        = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    markers    = cv2.watershed(bgr, markers)
    return len([u for u in np.unique(markers) if u > 1])


def _draw_filled(canvas: np.ndarray, contour, color, alpha: float = 0.35):
    """Semi-transparent filled contour drawn onto canvas in-place."""
    overlay = canvas.copy()
    cv2.drawContours(overlay, [contour], -1, color, -1)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)


def _classify_and_count(area: float, single_area: float,
                        comp_mask: np.ndarray,
                        ws_dist_thresh: float = 0.45):

    ratio = area / single_area
    if ratio < 1.6:
        return 1, 1
    elif ratio < 4.0:
        return max(2, round(ratio)), 2
    else:
        ws  = _watershed_split(comp_mask, dist_thresh=ws_dist_thresh)
        est = max(2, round(ratio * 0.9))   # slight downward bias
        return max(ws, est), 3



#  SCREW COUNTER
def count_screws(image_path: str,
                 debug_output: str   = None,
                 single_screw_area: int = 3506,
                 min_area: int          = 500,
                 border: int            = 30) -> int:
  
    raw = cv2.imread(image_path)
    if raw is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    img  = _add_border(raw, border)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu works well for screws (good contrast, no hollow centre)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = np.ones((3, 3), np.uint8)
    clean  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)
    clean  = cv2.morphologyEx(clean,  cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, 8)

    h_img, w_img = img.shape[:2]
    canvas = img.copy()
    count  = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x  = stats[i, cv2.CC_STAT_LEFT]
        y  = stats[i, cv2.CC_STAT_TOP]
        wb = stats[i, cv2.CC_STAT_WIDTH]
        hb = stats[i, cv2.CC_STAT_HEIGHT]

        # Reject blobs touching image border (vignetting / shadow artefacts)
        margin = 5
        if x < margin or y < margin or x+wb > w_img-margin or y+hb > h_img-margin:
            continue

        comp_mask = ((labels == i) * 255).astype(np.uint8)
        ctrs, _   = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

        n, tier = _classify_and_count(area, single_screw_area, comp_mask,
                                      ws_dist_thresh=0.45)
        count += n
        color  = (0, 255, 255)

        if ctrs:
            _draw_filled(canvas, ctrs[0], color, alpha=0.30)

    cv2.putText(canvas, f"COUNT: {count}",
                (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 220), 5)

    if debug_output:
        h_out = int(1000 * canvas.shape[0] / canvas.shape[1])
        cv2.imwrite(debug_output, cv2.resize(canvas, (1000, h_out)))
        print(f"saved → {debug_output}")

    return count



#  NUT COUNTER

def count_nuts(image_path: str, debug_output: str = None) -> int:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold (robust for lighting)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41, 2
    )

    #  Morphology 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    #  Connected components 
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, 8)

    canvas = img.copy()
    overlay = img.copy()

    count = 0

    h, w = img.shape[:2]

    # Estimating single nut area dynamically
    areas = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if 200 < area < 5000:  # filter reasonable nut sizes
            areas.append(area)

    if len(areas) == 0:
        print("No valid components found")
        return 0

    single_nut_area = np.median(areas)

    # Counting the nuts
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 100:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        wb = stats[i, cv2.CC_STAT_WIDTH]
        hb = stats[i, cv2.CC_STAT_HEIGHT]

        if x < 5 or y < 5 or x + wb > w - 5 or y + hb > h - 5:
            continue

        if area < 1.5 * single_nut_area:
            n = 1
   
        else:
            n = int(round(area / single_nut_area ))
   
        color = (0, 255, 255)
        count += n
        comp_mask = ((labels == i) * 255).astype(np.uint8)

        ctrs, _ = cv2.findContours(
            comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        color = (0, 255, 255)  # yellow

        if ctrs:
            cv2.drawContours(overlay, ctrs, -1, color, thickness=cv2.FILLED)


    canvas = cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0)
        

    cv2.putText(canvas, f"COUNT: {count}",
                (30, 70), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (0, 0, 255), 4)

    if debug_output:
        h_out = int(1000 * canvas.shape[0] / canvas.shape[1])
        cv2.imwrite(debug_output, cv2.resize(canvas, (1000, h_out)))
        print(f"saved -> {debug_output}")

    return count

#  Run this to find the correct single_screw_area / single_nut_area
def calibrate(image_path: str, mode: str = "screws"):
    img  = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if mode == "nuts":
        thresh = cv2.adaptiveThreshold(blur, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 31, 4)
        contours, hier = cv2.findContours(thresh, cv2.RETR_CCOMP,
                                          cv2.CHAIN_APPROX_SIMPLE)
        hier = hier[0]
        areas = [cv2.contourArea(c) for c, h in zip(contours, hier)
                 if h[2] != -1 and h[3] == -1]
    else:
        _, thresh = cv2.threshold(blur, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        k = np.ones((3,3), np.uint8)
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k, iterations=1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k, iterations=2)
        _, _, stats, _ = cv2.connectedComponentsWithStats(clean, 8)
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, len(stats))
                 if stats[i, cv2.CC_STAT_AREA] > 300]

    areas.sort()
    import statistics as st
    print(f"\n=== Calibration: {mode} ===")
    print(f"Total blobs found: {len(areas)}")
    if areas:
        print(f"Min area   : {areas[0]}")
        print(f"Median area: {st.median(areas):.0f}  ← likely single-object area")
        print(f"Max area   : {areas[-1]}")
        print(f"\nSmallest 10 areas: {areas[:10]}")



#  CLI

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python3 s_n.py screws  <image> [output.jpg]")
        print("  python3 s_n.py nuts    <image> [output.jpg]")
        print("  python3 s_n.py calibrate <image> [screws|nuts]")
        sys.exit(1)

    mode     = sys.argv[1].lower()
    img_path = sys.argv[2]
    extra    = sys.argv[3] if len(sys.argv) > 3 else None

    if mode == "screws":
        n = count_screws(img_path, debug_output=extra)
        print(f"Count: {n}")

    elif mode == "nuts":
        n = count_nuts(img_path, debug_output=extra)
        print(f"Count: {n}")

    elif mode == "calibrate":
        obj_mode = extra if extra in ("screws", "nuts") else "screws"
        calibrate(img_path, mode=obj_mode)

    else:
        print("Mode must be 'screws', 'nuts', or 'calibrate'")
        sys.exit(1)