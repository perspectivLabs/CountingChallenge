from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def zhang_suen_thinning(binary_mask: np.ndarray, max_iter: int = 120) -> np.ndarray:
    """
    Fast vectorized Zhang-Suen thinning in pure NumPy.
    Input: binary uint8 mask (0 or >0).
    Output: 1-pixel-wide skeleton uint8 mask (0 or 255).
    """
    img = (binary_mask > 0).astype(np.uint8)
    if img.shape[0] < 3 or img.shape[1] < 3 or not np.any(img):
        return np.zeros_like(binary_mask, dtype=np.uint8)

    # Pad by 1 pixel so border pixels are handled cleanly
    padded = np.pad(img, 1, mode="constant", constant_values=0)

    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p1 = padded[1:-1, 1:-1]
            p2 = padded[0:-2, 1:-1]  # N
            p3 = padded[0:-2, 2:]    # NE
            p4 = padded[1:-1, 2:]    # E
            p5 = padded[2:, 2:]      # SE
            p6 = padded[2:, 1:-1]    # S
            p7 = padded[2:, 0:-2]    # SW
            p8 = padded[1:-1, 0:-2]  # W
            p9 = padded[0:-2, 0:-2]  # NW

            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )

            cond_common = (p1 == 1) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1)
            if step == 0:
                cond_step = ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                cond_step = ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)

            to_remove = cond_common & cond_step
            if np.any(to_remove):
                padded[1:-1, 1:-1][to_remove] = 0
                changed = True

        if not changed:
            break

    return (padded[1:-1, 1:-1] * 255).astype(np.uint8)


def analyze_skeleton_topology(skel_mask: np.ndarray) -> tuple[int, int, int]:
    """Return (skeleton_pixels, num_endpoints, num_junction_clusters) on a 1px skeleton."""
    s = (skel_mask > 0).astype(np.uint8)
    total_px = int(np.sum(s))
    if total_px == 0:
        return 0, 0, 0

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    deg = cv2.filter2D(s, cv2.CV_8U, kernel, borderType=cv2.BORDER_CONSTANT)

    endpoints = (s == 1) & (deg == 1)
    junctions = (s == 1) & (deg >= 3)

    num_endpoints = int(np.sum(endpoints))
    # Cluster adjacent junction pixels so a single T/X intersection counts as 1 junction
    num_junc_cc, _ = cv2.connectedComponents(junctions.astype(np.uint8), connectivity=8)
    num_junctions = max(0, num_junc_cc - 1)
    return total_px, num_endpoints, num_junctions


def contour_circularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    if perim <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (perim * perim))


@dataclass
class ComponentFeatures:
    comp_id: int
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    outer_contour: np.ndarray
    hole_contours: list[np.ndarray]
    valid_hole_contours: list[np.ndarray]
    # 7 core geometric properties
    area: float
    filled_area: float
    aspect_ratio: float
    major_length: float
    minor_width: float
    circularity: float
    solidity: float
    num_holes: int
    max_dist: float
    mean_dist: float
    dist_peaks: int
    skeleton_length: float
    skeleton_endpoints: int
    skeleton_junctions: int
    # Auxiliary stats & classification
    mean_contrast: float
    case_label: str = "unclassified"  # isolated_screw | isolated_nut | touching_nuts | touching_screws | noise
    estimated_count: int = 1
    roi_mask: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.uint8), repr=False)
    roi_skeleton: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.uint8), repr=False)
    roi_dist: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32), repr=False)


@dataclass
class ScenePrototype:
    scene_type: str  # "nuts" | "screws" | "mixed"
    single_screw_area: float
    single_screw_length: float
    single_screw_shaft_width: float
    single_screw_head_radius: float
    single_nut_filled_area: float
    median_hole_area: float
    median_hole_radius: float
    screw_template: np.ndarray | None = None
    upright_template: np.ndarray | None = None


def _compute_roi_geometry(
    fg_mask: np.ndarray,
    outer_cnt: np.ndarray,
    hole_cnts: list[np.ndarray],
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray, np.ndarray, float, float, int, float, int, int]:
    """Compute tight ROI mask, distance transform, and skeleton metrics for one component."""
    h_img, w_img = fg_mask.shape
    x, y, bw, bh = cv2.boundingRect(outer_cnt)
    pad = 2
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w_img, x + bw + pad), min(h_img, y + bh + pad)
    bbox = (x0, y0, x1 - x0, y1 - y0)

    # Build exact component mask (outer filled minus its holes)
    roi_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    offset = np.array([[[x0, y0]]], dtype=outer_cnt.dtype)
    cv2.drawContours(roi_mask, [outer_cnt - offset], -1, 255, thickness=cv2.FILLED)
    if hole_cnts:
        shifted_holes = [hc - offset for hc in hole_cnts]
        cv2.drawContours(roi_mask, shifted_holes, -1, 0, thickness=cv2.FILLED)

    # Distance transform
    roi_dist = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
    fg_pixels = roi_dist[roi_mask > 0]
    max_dist = float(fg_pixels.max()) if fg_pixels.size > 0 else 0.0
    mean_dist = float(fg_pixels.mean()) if fg_pixels.size > 0 else 0.0

    # Local distance peaks
    if max_dist > 1.5:
        peak_k = max(5, int(round(max_dist * 0.9)) | 1)
        peak_k = min(peak_k, 51)
        d_smooth = cv2.GaussianBlur(roi_dist, (5, 5), 0)
        d_dil = cv2.dilate(d_smooth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (peak_k, peak_k)))
        peaks = (d_smooth == d_dil) & (d_smooth >= 0.48 * float(d_smooth.max()))
        n_pk, _ = cv2.connectedComponents(peaks.astype(np.uint8), connectivity=8)
        dist_peaks = max(0, n_pk - 1)
    else:
        dist_peaks = 1 if fg_pixels.size > 0 else 0

    # Skeletonization (downscale large cluster ROIs for speed, then preserve topological stats)
    roi_h, roi_w = roi_mask.shape
    max_side = max(roi_h, roi_w)
    skel_scale = 1.0 if max_side <= 320 else 320.0 / max_side
    if skel_scale < 1.0:
        sw = max(3, int(round(roi_w * skel_scale)))
        sh = max(3, int(round(roi_h * skel_scale)))
        small_mask = (cv2.resize(roi_mask, (sw, sh), interpolation=cv2.INTER_AREA) >= 128).astype(np.uint8) * 255
        skel_small = zhang_suen_thinning(small_mask)
        skel_px, skel_end, skel_junc = analyze_skeleton_topology(skel_small)
        skel_len = float(skel_px) / skel_scale
        roi_skel = cv2.resize(skel_small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    else:
        roi_skel = zhang_suen_thinning(roi_mask)
        skel_px, skel_end, skel_junc = analyze_skeleton_topology(roi_skel)
        skel_len = float(skel_px)

    return bbox, roi_mask, roi_skel, roi_dist, max_dist, mean_dist, dist_peaks, skel_len, skel_end, skel_junc


def _extract_aligned_template(comp: ComponentFeatures) -> np.ndarray | None:
    """Rotate an isolated screw ROI mask to horizontal (0 deg) for rotation-invariant matching."""
    mask = comp.roi_mask
    if mask.size == 0 or np.count_nonzero(mask) < 20:
        return None
    pts = np.column_stack(np.where(mask > 0))[:, ::-1].astype(np.float32)  # (x, y)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (rw, rh), angle = rect
    if rw < rh:
        angle += 90.0
    h, w = mask.shape
    diag = int(np.ceil(np.hypot(w, h))) + 6
    canvas = np.zeros((diag, diag), dtype=np.uint8)
    ox = (diag - w) // 2
    oy = (diag - h) // 2
    canvas[oy : oy + h, ox : ox + w] = mask
    mat = cv2.getRotationMatrix2D((ox + cx, oy + cy), angle, 1.0)
    rot = cv2.warpAffine(canvas, mat, (diag, diag), flags=cv2.INTER_NEAREST)
    ys, xs = np.where(rot > 0)
    if ys.size == 0:
        return None
    y0, y1 = max(0, int(ys.min()) - 2), min(diag, int(ys.max()) + 3)
    x0, x1 = max(0, int(xs.min()) - 2), min(diag, int(xs.max()) + 3)
    return rot[y0:y1, x0:x1]


def extract_and_classify_components(
    fg_mask: np.ndarray,
    contrast_map: np.ndarray,
) -> tuple[list[ComponentFeatures], ScenePrototype]:
    """
    Stage 2: Extract connected regions, compute all 7 geometric features,
    infer image-specific object prototypes, and classify every component into:
      - isolated_nut / isolated_screw (Case 1)
      - touching_nuts (Case 2)
      - touching_screws (Case 3)
    """
    h_img, w_img = fg_mask.shape
    min_outer_area = max(30.0, (h_img * w_img) * 0.000018)
    min_hole_area = max(12.0, (h_img * w_img) * 0.000004)

    contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        empty_proto = ScenePrototype("screws", 500.0, 40.0, 10.0, 10.0, 500.0, 80.0, 5.0)
        return [], empty_proto

    hier = hierarchy[0]
    outer_to_holes: dict[int, list[np.ndarray]] = {}
    for idx, h_info in enumerate(hier):
        parent = int(h_info[3])
        if parent == -1:
            outer_to_holes.setdefault(idx, [])
        else:
            outer_to_holes.setdefault(parent, []).append(contours[idx])

    raw_components: list[ComponentFeatures] = []
    comp_id = 1
    for outer_idx, hole_cnts in outer_to_holes.items():
        outer_cnt = contours[outer_idx]
        filled_area = float(cv2.contourArea(outer_cnt))
        if filled_area < min_outer_area:
            continue

        # Validate candidate internal circular holes (circularity + solidity + size)
        valid_holes: list[np.ndarray] = []
        for hc in hole_cnts:
            ha = float(cv2.contourArea(hc))
            if ha < min_hole_area:
                continue
            h_circ = contour_circularity(hc)
            h_hull = cv2.convexHull(hc)
            h_sol = ha / max(float(cv2.contourArea(h_hull)), 1e-6)
            if h_circ >= 0.62 and h_sol >= 0.80:
                valid_holes.append(hc)

        bbox, roi_mask, roi_skel, roi_dist, max_dist, mean_dist, dist_peaks, skel_len, skel_end, skel_junc = (
            _compute_roi_geometry(fg_mask, outer_cnt, hole_cnts)
        )
        area = float(np.count_nonzero(roi_mask))
        if area < min_outer_area * 0.7:
            continue

        x0, y0, bw, bh = bbox
        roi_contrast = contrast_map[y0 : y0 + bh, x0 : x0 + bw]
        mean_contrast = float(roi_contrast[roi_mask > 0].mean()) if np.any(roi_mask > 0) else 0.0

        rect = cv2.minAreaRect(outer_cnt)
        rw, rh = float(rect[1][0]), float(rect[1][1])
        major_len = max(rw, rh)
        minor_wid = max(min(rw, rh), 1e-6)
        aspect_ratio = major_len / minor_wid

        circ = contour_circularity(outer_cnt)
        hull = cv2.convexHull(outer_cnt)
        hull_area = max(float(cv2.contourArea(hull)), 1e-6)
        solidity = filled_area / hull_area

        raw_components.append(
            ComponentFeatures(
                comp_id=comp_id,
                bbox=bbox,
                outer_contour=outer_cnt,
                hole_contours=hole_cnts,
                valid_hole_contours=valid_holes,
                area=area,
                filled_area=filled_area,
                aspect_ratio=aspect_ratio,
                major_length=major_len,
                minor_width=minor_wid,
                circularity=circ,
                solidity=solidity,
                num_holes=len(valid_holes),
                max_dist=max_dist,
                mean_dist=mean_dist,
                dist_peaks=dist_peaks,
                skeleton_length=skel_len,
                skeleton_endpoints=skel_end,
                skeleton_junctions=skel_junc,
                mean_contrast=mean_contrast,
                roi_mask=roi_mask,
                roi_skeleton=roi_skel,
                roi_dist=roi_dist,
            )
        )
        comp_id += 1

    if not raw_components:
        empty_proto = ScenePrototype("screws", 500.0, 40.0, 10.0, 10.0, 500.0, 80.0, 5.0)
        return [], empty_proto

    # Filter out very faint background smudges relative to true metallic parts
    contrasts = np.array([c.mean_contrast for c in raw_components])
    high_contrast_ref = float(np.percentile(contrasts, 75))
    min_valid_contrast = max(24.0, 0.40 * high_contrast_ref)

    # --- Determine whether the scene contains nuts (consistent circular holes) ---
    # In a hex nut, hole_radius is comparable to wall thickness (hr / max_dist in [0.55, 2.3]),
    # whereas specular pinholes on metallic screw heads have hr / max_dist << 0.45.
    all_hole_areas: list[float] = []
    all_hole_radii: list[float] = []
    for c in raw_components:
        if c.mean_contrast < min_valid_contrast:
            continue
        for hc in c.valid_hole_contours:
            ha = float(cv2.contourArea(hc))
            (_, _), hr = cv2.minEnclosingCircle(hc)
            if 0.55 * max(c.max_dist, 1.0) <= hr <= 2.35 * max(c.max_dist, 1.0):
                # For isolated single-hole candidates, verify hole is a substantial fraction of object area
                if len(c.valid_hole_contours) == 1 and ha < 0.07 * c.filled_area:
                    continue
                all_hole_areas.append(ha)
                all_hole_radii.append(float(hr))

    # Re-filter each component's valid_hole_contours against consistent nut-hole scale
    is_nut_scene = False
    med_hole_area = 80.0
    med_hole_radius = 5.0
    if len(all_hole_areas) >= 5:
        med_hole_area = float(np.median(all_hole_areas))
        med_hole_radius = float(np.median(all_hole_radii))
        consistent_total = 0
        for c in raw_components:
            filtered_holes = []
            for hc in c.valid_hole_contours:
                ha = float(cv2.contourArea(hc))
                (_, _), hr = cv2.minEnclosingCircle(hc)
                if (
                    0.25 * med_hole_area <= ha <= 3.0 * med_hole_area
                    and 0.55 * max(c.max_dist, 1.0) <= hr <= 2.40 * max(c.max_dist, 1.0)
                ):
                    if len(c.valid_hole_contours) == 1 and ha < 0.07 * c.filled_area:
                        continue
                    filtered_holes.append(hc)
            c.valid_hole_contours = filtered_holes
            c.num_holes = len(filtered_holes)
            consistent_total += c.num_holes
        if consistent_total >= 5 and consistent_total >= 0.45 * len(raw_components):
            is_nut_scene = True
    if not is_nut_scene:
        # In pure screw images, fill tiny specular glints inside screw heads so DT & skeleton stay clean
        for c in raw_components:
            c.valid_hole_contours = []
            c.num_holes = 0
            if c.hole_contours:
                large_cavities = [
                    hc for hc in c.hole_contours if cv2.contourArea(hc) >= 0.12 * c.filled_area
                ]
                (
                    c.bbox,
                    c.roi_mask,
                    c.roi_skeleton,
                    c.roi_dist,
                    c.max_dist,
                    c.mean_dist,
                    c.dist_peaks,
                    c.skeleton_length,
                    c.skeleton_endpoints,
                    c.skeleton_junctions,
                ) = _compute_roi_geometry(fg_mask, c.outer_contour, large_cavities)
                c.area = float(np.count_nonzero(c.roi_mask))

    # --- Estimate single-nut prototype ---
    isolated_nut_candidates = [
        c
        for c in raw_components
        if c.mean_contrast >= min_valid_contrast
        and c.num_holes == 1
        and c.solidity >= 0.80
        and c.aspect_ratio <= 1.5
    ]
    if isolated_nut_candidates:
        single_nut_filled_area = float(np.median([c.filled_area for c in isolated_nut_candidates]))
    else:
        # Geometric relation: hex nut outer area is ~4.5-6x its circular inner hole area
        single_nut_filled_area = med_hole_area * 5.2

    # --- Estimate single-screw prototype from high-solidity isolated objects ---
    non_noise = [c for c in raw_components if c.mean_contrast >= min_valid_contrast]
    if not non_noise:
        non_noise = raw_components

    # Candidate isolated screws: compact/solid (not L/V/W merged clusters) and no nut holes
    screw_candidates = [
        c
        for c in non_noise
        if c.num_holes == 0 and c.solidity >= 0.56
    ]
    if not screw_candidates:
        screw_candidates = [c for c in non_noise if c.num_holes == 0]
    if not screw_candidates:
        screw_candidates = non_noise

    # Discard tiny dust outliers before estimating single screw size
    cand_areas = np.array([c.area for c in screw_candidates])
    upper_ref = float(np.percentile(cand_areas, 75))
    screw_candidates = [c for c in screw_candidates if c.area >= 0.18 * upper_ref]
    if not screw_candidates:
        screw_candidates = non_noise

    # Separate upright circular screws for optional upright template, but keep ALL high-solidity
    # single-object candidates (including stubby pan-head screws with AR ~ 1.1-1.3) in ref_pool!
    upright_cands = [
        c for c in screw_candidates if c.aspect_ratio < 1.25 and c.circularity >= 0.74 and c.solidity >= 0.84
    ]
    high_sol_cands = [c for c in screw_candidates if c.solidity >= 0.58]
    ref_pool = high_sol_cands if len(high_sol_cands) >= 2 else screw_candidates
    ref_areas = np.array([c.area for c in ref_pool])
    # Use lower-middle quantile so occasional compact pairs don't inflate 1x single-screw size
    q_idx = 30 if len(ref_pool) <= 20 else 40
    est_1x = float(np.percentile(ref_areas, q_idx))
    tight_singles = [c for c in ref_pool if 0.55 * est_1x <= c.area <= 1.45 * est_1x]
    if not tight_singles:
        tight_singles = ref_pool

    single_screw_area = float(np.median([c.area for c in tight_singles]))
    single_screw_length = float(np.median([c.major_length for c in tight_singles]))
    single_screw_head_radius = float(np.median([c.max_dist for c in tight_singles]))
    single_screw_shaft_width = float(np.median([2.0 * c.mean_dist for c in tight_singles]))

    # Build representative screw template from the single screw closest to median area & length
    best_screw = min(
        tight_singles,
        key=lambda c: abs(c.area - single_screw_area) / max(single_screw_area, 1.0)
        + abs(c.major_length - single_screw_length) / max(single_screw_length, 1.0),
    )
    screw_template = _extract_aligned_template(best_screw) if not is_nut_scene else None

    upright_template = None
    if upright_cands and not is_nut_scene:
        med_up_area = float(np.median([c.area for c in upright_cands]))
        best_up = min(upright_cands, key=lambda c: abs(c.area - med_up_area))
        upright_template = _extract_aligned_template(best_up)

    # --- Final 3-Case Classification of Every Connected Component ---
    min_obj_area = (
        max(min_outer_area, 0.30 * single_nut_filled_area)
        if is_nut_scene
        else max(min_outer_area, 0.25 * single_screw_area)
    )

    for c in raw_components:
        if c.mean_contrast < min_valid_contrast or c.area < min_obj_area:
            c.case_label = "noise"
            c.estimated_count = 0
            continue

        if is_nut_scene:
            if c.num_holes >= 2:
                c.case_label = "touching_nuts"
                c.estimated_count = max(c.num_holes, int(round(c.filled_area / max(single_nut_filled_area, 1.0))))
            elif c.num_holes == 1 and c.filled_area <= 1.65 * single_nut_filled_area:
                c.case_label = "isolated_nut"
                c.estimated_count = 1
            elif c.filled_area > 1.65 * single_nut_filled_area:
                c.case_label = "touching_nuts"
                c.estimated_count = max(2, int(round(c.filled_area / max(single_nut_filled_area, 1.0))))
            else:
                # Single nut whose hole was barely occluded or small
                c.case_label = "isolated_nut"
                c.estimated_count = 1
        else:
            # Screw scene: distinguish isolated screws vs touching/overlapping screw clusters
            area_ratio = c.area / max(single_screw_area, 1.0)
            len_ratio = c.major_length / max(single_screw_length, 1.0)
            is_merged = (
                area_ratio >= 1.50
                or (area_ratio >= 1.22 and c.solidity < 0.56)
                or (area_ratio >= 1.28 and len_ratio >= 1.38)
                or (area_ratio >= 1.25 and c.skeleton_junctions >= 3)
            )
            if is_merged:
                c.case_label = "touching_screws"
                c.estimated_count = max(2, int(round(area_ratio)))
            else:
                c.case_label = "isolated_screw"
                c.estimated_count = 1

    scene_type = "nuts" if is_nut_scene else "screws"
    prototype = ScenePrototype(
        scene_type=scene_type,
        single_screw_area=single_screw_area,
        single_screw_length=single_screw_length,
        single_screw_shaft_width=single_screw_shaft_width,
        single_screw_head_radius=single_screw_head_radius,
        single_nut_filled_area=single_nut_filled_area,
        median_hole_area=med_hole_area,
        median_hole_radius=med_hole_radius,
        screw_template=screw_template,
        upright_template=upright_template,
    )
    return raw_components, prototype
