from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .features import ComponentFeatures, ScenePrototype, zhang_suen_thinning
from .nuts import InstanceMask


def _mask_to_instance(
    sub_mask: np.ndarray,
    x0: int,
    y0: int,
    inst_id: int,
    kind: str,
    debug_info: dict | None = None,
) -> InstanceMask | None:
    ys, xs = np.where(sub_mask > 0)
    if ys.size == 0:
        return None
    iy0, iy1 = int(ys.min()), int(ys.max()) + 1
    ix0, ix1 = int(xs.min()), int(xs.max()) + 1
    tight = sub_mask[iy0:iy1, ix0:ix1].copy()
    cx = int(round(x0 + xs.mean()))
    cy = int(round(y0 + ys.mean()))
    return InstanceMask(
        instance_id=inst_id,
        bbox=(x0 + ix0, y0 + iy0, ix1 - ix0, iy1 - iy0),
        mask=tight,
        center=(cx, cy),
        kind=kind,
        debug_info=debug_info or {},
    )


def _partition_roi_by_seeds(
    roi_mask: np.ndarray,
    seed_masks: list[np.ndarray],
    x0: int,
    y0: int,
    min_inst_area: float,
    start_id: int,
    kind: str,
) -> list[InstanceMask]:
    """
    Partition roi_mask into disjoint per-instance masks using marker-controlled watershed
    on distance-from-seeds topography restricted to roi_mask.
    """
    if not seed_masks:
        inst = _mask_to_instance(roi_mask, x0, y0, start_id, kind)
        return [inst] if inst else []

    h, w = roi_mask.shape
    markers = np.zeros((h, w), dtype=np.int32)
    markers[roi_mask == 0] = 1

    valid_labels: list[int] = []
    for idx, sm in enumerate(seed_masks, start=2):
        # Keep only pixels inside roi_mask that aren't already claimed by a prior seed
        s_clean = (sm > 0) & (roi_mask > 0) & (markers == 0)
        if np.any(s_clean):
            markers[s_clean] = idx
            valid_labels.append(idx)

    if len(valid_labels) <= 1:
        inst = _mask_to_instance(roi_mask, x0, y0, start_id, kind)
        return [inst] if inst else []

    # Use strictly convex quadratic distance surface so cv2.watershed expands all seeds
    # symmetrically (step gradient ~ r) without scanline tie-breaking bias.
    seed_bin = (markers >= 2).astype(np.uint8)
    dist_from_seeds = cv2.distanceTransform(1 - seed_bin, cv2.DIST_L2, 5)
    max_d = max(float(dist_from_seeds[roi_mask > 0].max()), 1.0)
    quad_surface = np.clip((dist_from_seeds / max_d) ** 2 * 255.0, 0, 255).astype(np.uint8)
    surface_bgr = cv2.cvtColor(quad_surface, cv2.COLOR_GRAY2BGR)
    cv2.watershed(surface_bgr, markers)

    # Assign any 1px watershed dam pixels (-1) inside roi_mask to nearest valid neighbor
    dam = (markers == -1) & (roi_mask > 0)
    if np.any(dam):
        dil = cv2.dilate(np.where(markers >= 2, markers, 0).astype(np.uint16), np.ones((3, 3), np.uint8))
        markers[dam] = dil[dam].astype(np.int32)

    instances: list[InstanceMask] = []
    next_id = start_id
    for lbl in valid_labels:
        sub = ((markers == lbl) & (roi_mask > 0)).astype(np.uint8) * 255
        if np.count_nonzero(sub) < min_inst_area:
            continue
        inst = _mask_to_instance(sub, x0, y0, next_id, kind)
        if inst is not None:
            instances.append(inst)
            next_id += 1

    if not instances:
        inst = _mask_to_instance(roi_mask, x0, y0, start_id, kind)
        return [inst] if inst else []
    return instances


# =============================================================================
# Technique 3A: Skeleton Graph Analysis
# =============================================================================

def _trace_ordered_path(coords: np.ndarray) -> np.ndarray:
    """Order an 8-connected 1px branch into a polyline from one endpoint to the other."""
    if len(coords) <= 2:
        return coords
    pts_set = {(int(r), int(c)) for r, c in coords}
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in pts_set:
        nbrs = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in pts_set:
                    nbrs.append(nb)
        adj[(r, c)] = nbrs

    ends = [p for p, nbrs in adj.items() if len(nbrs) <= 1]
    start = ends[0] if ends else next(iter(pts_set))

    ordered = [start]
    visited = {start}
    curr = start
    while True:
        next_cands = [nb for nb in adj[curr] if nb not in visited]
        if not next_cands:
            break
        curr = next_cands[0]
        visited.add(curr)
        ordered.append(curr)
    return np.array(ordered, dtype=np.int32)


def _branch_tangent(path: np.ndarray, at_start: bool, span: int = 8) -> np.ndarray:
    """Unit tangent vector pointing OUTWARD from the specified end of an ordered (r, c) path."""
    if len(path) < 2:
        return np.array([1.0, 0.0], dtype=np.float32)
    if at_start:
        p0 = path[0].astype(np.float32)
        p1 = path[min(len(path) - 1, span)].astype(np.float32)
    else:
        p0 = path[-1].astype(np.float32)
        p1 = path[max(0, len(path) - 1 - span)].astype(np.float32)
    vec = p1 - p0
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)


def _smooth_and_skeletonize(work_mask: np.ndarray, prune_iters: int = 4) -> np.ndarray:
    """
    Smooth jagged screw-thread sawtooths before Zhang-Suen thinning and prune tiny
    micro-spurs so shafts produce clean, uninterrupted centerlines.
    """
    # Smooth sawtooth threads without merging separated gaps
    smooth = cv2.GaussianBlur(work_mask, (5, 5), 0)
    clean_mask = (smooth >= 128).astype(np.uint8) * 255
    if not np.any(clean_mask):
        clean_mask = work_mask
    skel = zhang_suen_thinning(clean_mask)

    # Iteratively prune short micro-spurs caused by thread bumps
    s = (skel > 0).astype(np.uint8)
    deg_k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    for _ in range(prune_iters):
        deg = cv2.filter2D(s, cv2.CV_8U, deg_k, borderType=cv2.BORDER_CONSTANT)
        endpoints = (s == 1) & (deg == 1)
        if not np.any(endpoints):
            break
        # Only prune endpoints that are within 1 step of a junction or short spur
        # (protects genuine long shafts while stripping tiny 1-4px thread teeth)
        s[endpoints] = 0
    return (s * 255).astype(np.uint8)


def _crossing_number(s_bin: np.ndarray) -> np.ndarray:
    """
    Compute topological Crossing Number CN = 0.5 * sum |p_i - p_{i+1}| around 8-neighbors.
    CN == 1 -> endpoint, CN == 2 -> interior line pixel (even on diagonal staircases!),
    CN >= 3 -> true topological junction.
    """
    padded = np.pad(s_bin.astype(np.int16), 1, mode="constant", constant_values=0)
    ring = [
        padded[0:-2, 1:-1],  # N
        padded[0:-2, 2:],    # NE
        padded[1:-1, 2:],    # E
        padded[2:, 2:],      # SE
        padded[2:, 1:-1],    # S
        padded[2:, 0:-2],    # SW
        padded[1:-1, 0:-2],  # W
        padded[0:-2, 0:-2],  # NW
    ]
    transitions = np.zeros_like(s_bin, dtype=np.int16)
    for i in range(8):
        transitions += np.abs(ring[i] - ring[(i + 1) % 8])
    return (transitions // 2).astype(np.uint8)


def separate_skeleton_graph(
    comp: ComponentFeatures,
    proto: ScenePrototype,
    start_id: int = 1,
) -> tuple[list[InstanceMask], dict]:
    """
    Technique 3A: Skeleton Graph Analysis.
    Smooths thread noise, extracts 1px skeleton, detects true junctions via Crossing Number,
    decomposes into branches, prunes screw-head ears, merges colinear crossing branches,
    splits colinear end-to-end chains at distance minima, locates heads vs tips,
    and partitions the cluster mask.
    """
    x0, y0, bw, bh = comp.bbox
    roi_mask = comp.roi_mask
    roi_dist = comp.roi_dist

    # Work at a clean scale where single screw length is ~65px for crisp topology
    target_len = 65.0
    scale = min(1.0, target_len / max(proto.single_screw_length, 20.0))
    scale = max(scale, 0.20)
    sw = max(5, int(round(bw * scale)))
    sh = max(5, int(round(bh * scale)))

    if scale < 0.99:
        work_mask = (cv2.resize(roi_mask, (sw, sh), interpolation=cv2.INTER_AREA) >= 128).astype(np.uint8) * 255
        work_dist = cv2.distanceTransform(work_mask, cv2.DIST_L2, 5)
    else:
        work_mask = roi_mask
        work_dist = roi_dist

    skel = _smooth_and_skeletonize(work_mask, prune_iters=4)
    s_bin = (skel > 0).astype(np.uint8)
    cn = _crossing_number(s_bin)

    junction_mask = ((s_bin == 1) & (cn >= 3)).astype(np.uint8)
    # Dilate junctions by 1px to cleanly separate incident branches
    junc_dil = cv2.dilate(junction_mask, np.ones((3, 3), np.uint8))
    num_junc, junc_labels = cv2.connectedComponents(junc_dil, connectivity=8)

    branch_mask = ((s_bin == 1) & (junc_dil == 0)).astype(np.uint8)
    num_br, br_labels = cv2.connectedComponents(branch_mask, connectivity=8)

    s_len_ref = proto.single_screw_length * scale
    s_head_r = proto.single_screw_head_radius * scale
    spur_max_len = max(5.0, min(0.32 * s_len_ref, 1.50 * s_head_r))

    # Extract branches with their attached junction IDs
    branches: list[dict] = []
    for b_id in range(1, num_br):
        coords = np.column_stack(np.where(br_labels == b_id))
        if len(coords) < 2:
            continue
        path = _trace_ordered_path(coords)
        length = float(len(path))

        # Check which junctions each endpoint touches
        def _attached_junc(pt_rc: np.ndarray) -> int:
            r, c = int(pt_rc[0]), int(pt_rc[1])
            r0, r1 = max(0, r - 2), min(sh, r + 3)
            c0, c1 = max(0, c - 2), min(sw, c + 3)
            vals = junc_labels[r0:r1, c0:c1]
            nz = vals[vals > 0]
            return int(nz[0]) if nz.size > 0 else 0

        j_start = _attached_junc(path[0])
        j_end = _attached_junc(path[-1])
        is_free_spur = ((j_start > 0) != (j_end > 0)) and (length < spur_max_len)
        if is_free_spur:
            continue

        branches.append(
            {
                "id": b_id,
                "path": path,
                "length": length,
                "j_start": j_start,
                "j_end": j_end,
            }
        )

    # Merge colinear branch pairs at junctions (representing crossing screws)
    merged_flags = [False] * len(branches)
    axes_paths: list[np.ndarray] = []

    junc_to_inc: dict[int, list[tuple[int, bool]]] = {}
    for idx, br in enumerate(branches):
        if br["j_start"] > 0:
            junc_to_inc.setdefault(br["j_start"], []).append((idx, True))
        if br["j_end"] > 0:
            junc_to_inc.setdefault(br["j_end"], []).append((idx, False))

    for j_id, inc in junc_to_inc.items():
        if len(inc) < 2:
            continue
        # Evaluate all pairs at this junction for colinearity
        pair_scores: list[tuple[float, int, bool, int, bool]] = []
        for i in range(len(inc)):
            idx1, at_s1 = inc[i]
            if merged_flags[idx1]:
                continue
            t1 = _branch_tangent(branches[idx1]["path"], at_s1)
            for j in range(i + 1, len(inc)):
                idx2, at_s2 = inc[j]
                if merged_flags[idx2] or idx1 == idx2:
                    continue
                t2 = _branch_tangent(branches[idx2]["path"], at_s2)
                # Colinear across a junction means outward tangents point in opposite directions (dot ~ -1)
                dot = float(np.dot(t1, t2))
                comb_len = branches[idx1]["length"] + branches[idx2]["length"]
                if dot <= -0.82 and comb_len <= 1.40 * s_len_ref:
                    pair_scores.append((dot, idx1, at_s1, idx2, at_s2))

        pair_scores.sort(key=lambda t: t[0])
        for _, idx1, at_s1, idx2, at_s2 in pair_scores:
            if merged_flags[idx1] or merged_flags[idx2]:
                continue
            p1 = branches[idx1]["path"]
            p2 = branches[idx2]["path"]
            if at_s1:
                p1 = p1[::-1]
            if not at_s2:
                p2 = p2[::-1]
            axes_paths.append(np.vstack([p1, p2]))
            merged_flags[idx1] = True
            merged_flags[idx2] = True

    for idx, br in enumerate(branches):
        if not merged_flags[idx]:
            axes_paths.append(br["path"])

    # Split overly long chains (colinear end-to-end touching screws) at local distance minima
    final_paths: list[np.ndarray] = []
    for path in axes_paths:
        if len(path) >= int(1.50 * s_len_ref):
            n_splits = max(2, int(round(len(path) / max(s_len_ref, 10.0))))
            # Find cut indices near expected intervals with lowest distance transform (narrowest pinch)
            cut_indices = [0]
            for k in range(1, n_splits):
                est = int(round(k * len(path) / n_splits))
                win = max(3, int(0.20 * len(path) / n_splits))
                lo = max(cut_indices[-1] + 3, est - win)
                hi = min(len(path) - 3, est + win)
                if hi > lo:
                    prof = np.array([work_dist[r, c] for r, c in path[lo:hi]])
                    best_cut = lo + int(np.argmin(prof))
                    cut_indices.append(best_cut)
            cut_indices.append(len(path))
            for a, b in zip(cut_indices[:-1], cut_indices[1:]):
                if b - a >= max(4, int(0.25 * s_len_ref)):
                    final_paths.append(path[a:b])
        elif len(path) >= max(4, int(0.28 * s_len_ref)):
            final_paths.append(path)

    # Convert each recovered axis path into a full-resolution seed mask & locate head vs tip
    seed_masks: list[np.ndarray] = []
    debug_axes: list[np.ndarray] = []
    debug_heads: list[tuple[int, int]] = []
    debug_tips: list[tuple[int, int]] = []

    inv_scale = 1.0 / scale
    for path in final_paths:
        # Trim 15% from ends so adjacent seeds at a junction don't touch each other
        trim = max(1, int(0.12 * len(path)))
        core = path[trim : max(trim + 1, len(path) - trim)]

        # Determine head (larger distance transform) vs shaft tip
        d_start = float(work_dist[path[0, 0], path[0, 1]])
        d_end = float(work_dist[path[-1, 0], path[-1, 1]])
        head_rc = path[0] if d_start >= d_end else path[-1]
        tip_rc = path[-1] if d_start >= d_end else path[0]

        debug_heads.append((int(round(x0 + head_rc[1] * inv_scale)), int(round(y0 + head_rc[0] * inv_scale))))
        debug_tips.append((int(round(x0 + tip_rc[1] * inv_scale)), int(round(y0 + tip_rc[0] * inv_scale))))

        pts_full = np.column_stack(
            [np.round(path[:, 1] * inv_scale), np.round(path[:, 0] * inv_scale)]
        ).astype(np.int32)
        debug_axes.append(pts_full + np.array([[x0, y0]], dtype=np.int32))

        sm = np.zeros_like(roi_mask)
        core_xy = np.column_stack(
            [np.round(core[:, 1] * inv_scale), np.round(core[:, 0] * inv_scale)]
        ).astype(np.int32)
        if len(core_xy) >= 2:
            cv2.polylines(sm, [core_xy.reshape(-1, 1, 2)], False, 255, thickness=max(2, int(round(inv_scale))))
        else:
            cv2.circle(sm, (int(core_xy[0, 0]), int(core_xy[0, 1])), max(2, int(round(inv_scale))), 255, -1)
        seed_masks.append(sm)

    min_inst_area = max(25.0, 0.28 * proto.single_screw_area)
    instances = _partition_roi_by_seeds(
        roi_mask, seed_masks, x0, y0, min_inst_area, start_id, "screw_skeleton_graph"
    )
    debug_info = {
        "axes": debug_axes,
        "heads": debug_heads,
        "tips": debug_tips,
        "num_branches": len(final_paths),
    }
    return instances, debug_info


# =============================================================================
# Technique 3B: Hough Line Detection
# =============================================================================

def _merge_hough_segments(
    segments: list[tuple[float, float, float, float]],
    max_len: float,
    perp_tol: float,
    gap_tol: float,
) -> list[tuple[float, float, float, float]]:
    """Merge colinear, nearby overlapping Hough segments into distinct screw centerlines."""
    if not segments:
        return []

    # Sort longest first
    def seg_len(s: tuple[float, float, float, float]) -> float:
        return float(np.hypot(s[2] - s[0], s[3] - s[1]))

    segs = sorted(segments, key=seg_len, reverse=True)
    merged: list[tuple[float, float, float, float]] = []

    for x1, y1, x2, y2 in segs:
        vx, vy = x2 - x1, y2 - y1
        L = float(np.hypot(vx, vy))
        if L < 1e-3:
            continue
        ang = float(np.degrees(np.arctan2(vy, vx))) % 180.0
        mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

        did_merge = False
        for i, (ax1, ay1, ax2, ay2) in enumerate(merged):
            avx, avy = ax2 - ax1, ay2 - ay1
            aL = float(np.hypot(avx, avy))
            a_ang = float(np.degrees(np.arctan2(avy, avx))) % 180.0
            d_ang = min(abs(ang - a_ang), 180.0 - abs(ang - a_ang))
            if d_ang > 20.0:
                continue

            # Perpendicular distance of midpoint to existing line
            ux, uy = avx / aL, avy / aL
            perp = abs((mx - ax1) * (-uy) + (my - ay1) * ux)
            if perp > perp_tol:
                continue

            # Project all 4 endpoints onto existing line direction
            projs = [
                0.0,
                aL,
                (x1 - ax1) * ux + (y1 - ay1) * uy,
                (x2 - ax1) * ux + (y2 - ay1) * uy,
            ]
            p_min, p_max = min(projs), max(projs)
            seg_pmin = min(projs[2], projs[3])
            seg_pmax = max(projs[2], projs[3])
            gap = max(0.0, seg_pmin - aL, -seg_pmax)
            if gap <= gap_tol and (p_max - p_min) <= max_len:
                nx1 = ax1 + p_min * ux
                ny1 = ay1 + p_min * uy
                nx2 = ax1 + p_max * ux
                ny2 = ay1 + p_max * uy
                merged[i] = (nx1, ny1, nx2, ny2)
                did_merge = True
                break

        if not did_merge:
            merged.append((x1, y1, x2, y2))

    return merged


def separate_hough_lines(
    comp: ComponentFeatures,
    proto: ScenePrototype,
    start_id: int = 1,
) -> tuple[list[InstanceMask], dict]:
    """
    Technique 3B: Hough Line Detection on skeletonized cluster centerlines.
    Detects screw shafts via Probabilistic Hough Transform, merges colinear fragments,
    rescues any non-elongated upright screws in the cluster, and partitions the mask.
    """
    x0, y0, bw, bh = comp.bbox
    roi_mask = comp.roi_mask

    target_len = 65.0
    scale = min(1.0, target_len / max(proto.single_screw_length, 20.0))
    scale = max(scale, 0.20)
    sw = max(5, int(round(bw * scale)))
    sh = max(5, int(round(bh * scale)))

    if scale < 0.99:
        work_mask = (cv2.resize(roi_mask, (sw, sh), interpolation=cv2.INTER_AREA) >= 128).astype(np.uint8) * 255
    else:
        work_mask = roi_mask

    skel = _smooth_and_skeletonize(work_mask, prune_iters=3)
    s_len_ref = proto.single_screw_length * scale
    s_head_r = proto.single_screw_head_radius * scale

    min_line_len = max(8, int(round(0.32 * s_len_ref)))
    max_line_gap = max(3, int(round(0.16 * s_len_ref)))
    hough_thresh = max(6, int(round(0.22 * s_len_ref)))

    raw_lines = cv2.HoughLinesP(
        skel,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=hough_thresh,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )

    segments: list[tuple[float, float, float, float]] = []
    if raw_lines is not None:
        for line in raw_lines.reshape(-1, 4):
            segments.append((float(line[0]), float(line[1]), float(line[2]), float(line[3])))

    merged = _merge_hough_segments(
        segments,
        max_len=1.35 * s_len_ref,
        perp_tol=max(4.0, 0.75 * s_head_r),
        gap_tol=max(4.0, 0.22 * s_len_ref),
    )

    inv_scale = 1.0 / scale
    seed_masks: list[np.ndarray] = []
    debug_lines: list[tuple[tuple[int, int], tuple[int, int]]] = []
    covered_work = np.zeros_like(work_mask)

    for lx1, ly1, lx2, ly2 in merged:
        # Trim 12% from each end to prevent touching seeds at crossings
        dx, dy = lx2 - lx1, ly2 - ly1
        tx1, ty1 = lx1 + 0.12 * dx, ly1 + 0.12 * dy
        tx2, ty2 = lx2 - 0.12 * dx, ly2 - 0.12 * dy

        p1_full = (int(round(lx1 * inv_scale)), int(round(ly1 * inv_scale)))
        p2_full = (int(round(lx2 * inv_scale)), int(round(ly2 * inv_scale)))
        debug_lines.append(((x0 + p1_full[0], y0 + p1_full[1]), (x0 + p2_full[0], y0 + p2_full[1])))

        sm = np.zeros_like(roi_mask)
        cv2.line(
            sm,
            (int(round(tx1 * inv_scale)), int(round(ty1 * inv_scale))),
            (int(round(tx2 * inv_scale)), int(round(ty2 * inv_scale))),
            255,
            thickness=max(2, int(round(inv_scale))),
        )
        seed_masks.append(sm)
        cv2.line(
            covered_work,
            (int(round(lx1)), int(round(ly1))),
            (int(round(lx2)), int(round(ly2))),
            255,
            thickness=max(5, int(round(2.4 * s_head_r))),
        )

    # Rescue any large uncovered compact sub-region (e.g. upright circular screw touching a shaft)
    uncovered = cv2.bitwise_and(work_mask, cv2.bitwise_not(covered_work))
    min_uncov = max(15.0, 0.40 * proto.single_screw_area * (scale * scale))
    n_cc, cc_lbl, cc_stats, cc_cent = cv2.connectedComponentsWithStats(uncovered, connectivity=8)
    for lbl in range(1, n_cc):
        if cc_stats[lbl, cv2.CC_STAT_AREA] >= min_uncov:
            cx_f = int(round(cc_cent[lbl][0] * inv_scale))
            cy_f = int(round(cc_cent[lbl][1] * inv_scale))
            sm = np.zeros_like(roi_mask)
            cv2.circle(sm, (cx_f, cy_f), max(3, int(round(0.45 * proto.single_screw_head_radius))), 255, -1)
            seed_masks.append(sm)

    min_inst_area = max(25.0, 0.28 * proto.single_screw_area)
    instances = _partition_roi_by_seeds(
        roi_mask, seed_masks, x0, y0, min_inst_area, start_id, "screw_hough_lines"
    )
    return instances, {"lines": debug_lines, "num_lines": len(merged)}


# =============================================================================
# Technique 3C: Rotation-Invariant Template Matching
# =============================================================================

def _rotate_template(tpl: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = tpl.shape
    diag = int(np.ceil(np.hypot(w, h))) + 4
    canvas = np.zeros((diag, diag), dtype=np.uint8)
    ox, oy = (diag - w) // 2, (diag - h) // 2
    canvas[oy : oy + h, ox : ox + w] = tpl
    mat = cv2.getRotationMatrix2D((diag / 2.0, diag / 2.0), angle_deg, 1.0)
    rot = cv2.warpAffine(canvas, mat, (diag, diag), flags=cv2.INTER_NEAREST)
    ys, xs = np.where(rot > 0)
    if ys.size == 0:
        return tpl
    return rot[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]


def build_template_bank(proto: ScenePrototype, target_len: float = 44.0) -> tuple[list[dict], float]:
    """Precompute rotated distance-transform templates for fast rotation-invariant matching."""
    if proto.screw_template is None:
        return [], 1.0

    scale = min(1.0, target_len / max(proto.single_screw_length, 15.0))
    scale = max(scale, 0.15)
    th, tw = proto.screw_template.shape
    sw = max(5, int(round(tw * scale)))
    sh = max(5, int(round(th * scale)))
    base_tpl = (cv2.resize(proto.screw_template, (sw, sh), interpolation=cv2.INTER_AREA) >= 128).astype(np.uint8) * 255

    bank: list[dict] = []
    for angle in range(0, 360, 15):
        rot_mask = _rotate_template(base_tpl, float(angle))
        rot_dt = cv2.distanceTransform(rot_mask, cv2.DIST_L2, 5)
        if float(rot_dt.max()) <= 0:
            continue
        rot_dt = rot_dt / float(rot_dt.max())
        bank.append(
            {
                "angle": angle,
                "kind": "lying",
                "mask": rot_mask,
                "dt": rot_dt.astype(np.float32),
                "area": float(np.count_nonzero(rot_mask)),
            }
        )

    if proto.upright_template is not None:
        uh, uw = proto.upright_template.shape
        usw = max(5, int(round(uw * scale)))
        ush = max(5, int(round(uh * scale)))
        up_mask = (
            cv2.resize(proto.upright_template, (usw, ush), interpolation=cv2.INTER_AREA) >= 128
        ).astype(np.uint8) * 255
        up_dt = cv2.distanceTransform(up_mask, cv2.DIST_L2, 5)
        if float(up_dt.max()) > 0:
            bank.append(
                {
                    "angle": 0,
                    "kind": "upright",
                    "mask": up_mask,
                    "dt": (up_dt / float(up_dt.max())).astype(np.float32),
                    "area": float(np.count_nonzero(up_mask)),
                }
            )

    return bank, scale


def separate_template_matching(
    comp: ComponentFeatures,
    proto: ScenePrototype,
    template_bank: list[dict],
    bank_scale: float,
    start_id: int = 1,
) -> tuple[list[InstanceMask], dict]:
    """
    Technique 3C: Rotation-Invariant Template Matching (iterative matching pursuit).
    Matches rotated distance-transform templates across 24 orientations (15-deg steps)
    plus optional upright screw template, greedily explaining the cluster mask.
    """
    x0, y0, bw, bh = comp.bbox
    roi_mask = comp.roi_mask

    if not template_bank:
        inst = _mask_to_instance(roi_mask, x0, y0, start_id, "screw_template_matching")
        return ([inst] if inst else []), {"matches": []}

    scale = bank_scale
    sw = max(5, int(round(bw * scale)))
    sh = max(5, int(round(bh * scale)))
    work_mask = (cv2.resize(roi_mask, (sw, sh), interpolation=cv2.INTER_AREA) >= 128).astype(np.uint8) * 255

    # Pad ROI so templates can match screws right up to the ROI border
    max_tpl_dim = max(max(t["mask"].shape) for t in template_bank)
    pad = max_tpl_dim // 2 + 4
    padded_mask = cv2.copyMakeBorder(work_mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    remaining = padded_mask.copy()

    # Fixed normalized distance transform of the cluster for shape scoring
    base_dt = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 5)
    dt_scale = max(float(base_dt.max()), 1.0)
    base_dt_norm = (base_dt / dt_scale).astype(np.float32)

    ref_tpl_area = float(np.median([t["area"] for t in template_bank if t["kind"] == "lying"]))
    max_matches = max(2, int(np.ceil(comp.area / max(0.58 * proto.single_screw_area, 1.0))) + 1)

    matched_items: list[dict] = []
    occupied_centers: list[tuple[float, float]] = []
    min_center_sep = max(6.0, 0.38 * proto.single_screw_length * scale)

    for _ in range(max_matches):
        rem_area = float(np.count_nonzero(remaining))
        if rem_area < 0.38 * ref_tpl_area:
            break

        # Compute DT of remaining mask blended with original DT so crossed screws remain matchable
        rem_dt = cv2.distanceTransform(remaining, cv2.DIST_L2, 5)
        rem_dt_norm = (rem_dt / dt_scale).astype(np.float32)
        search_dt = 0.65 * rem_dt_norm + 0.35 * base_dt_norm

        best_score = -1.0
        best_cand: dict | None = None

        for tpl in template_bank:
            th, tw = tpl["mask"].shape
            if th > search_dt.shape[0] or tw > search_dt.shape[1]:
                continue
            res = cv2.matchTemplate(search_dt, tpl["dt"], cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val < 0.25:
                continue

            mx, my = max_loc
            cx_p, cy_p = mx + tw * 0.5, my + th * 0.5
            if any(np.hypot(cx_p - ox, cy_p - oy) < min_center_sep for ox, oy in occupied_centers):
                continue

            patch_rem = remaining[my : my + th, mx : mx + tw]
            patch_all = padded_mask[my : my + th, mx : mx + tw]
            tpl_bin = tpl["mask"] > 0

            ov_rem = float(np.count_nonzero(tpl_bin & (patch_rem > 0))) / max(tpl["area"], 1.0)
            ov_all = float(np.count_nonzero(tpl_bin & (patch_all > 0))) / max(tpl["area"], 1.0)
            if ov_rem < 0.36 or ov_all < 0.52:
                continue

            score = 0.50 * float(max_val) + 0.35 * ov_rem + 0.15 * ov_all
            if score > best_score:
                best_score = score
                best_cand = {
                    "tpl": tpl,
                    "mx": mx,
                    "my": my,
                    "cx": cx_p,
                    "cy": cy_p,
                    "ncc": float(max_val),
                    "ov_rem": ov_rem,
                    "score": score,
                }

        if best_cand is None or best_score < 0.42:
            break

        matched_items.append(best_cand)
        occupied_centers.append((best_cand["cx"], best_cand["cy"]))

        # Suppress explained pixels from remaining mask
        tpl = best_cand["tpl"]
        mx, my = best_cand["mx"], best_cand["my"]
        th, tw = tpl["mask"].shape
        remaining[my : my + th, mx : mx + tw][tpl["mask"] > 0] = 0

    inv_scale = 1.0 / scale
    seed_masks: list[np.ndarray] = []
    debug_matches: list[dict] = []

    for m in matched_items:
        tpl = m["tpl"]
        mx, my = m["mx"], m["my"]
        th, tw = tpl["mask"].shape

        # Erode template slightly to get a clean non-overlapping core seed
        core = cv2.erode(tpl["mask"], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        if np.count_nonzero(core) < 5:
            core = tpl["mask"]

        work_seed = np.zeros_like(padded_mask)
        work_seed[my : my + th, mx : mx + tw] = core
        unpadded = work_seed[pad : pad + sh, pad : pad + sw]
        full_seed = (cv2.resize(unpadded, (bw, bh), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8) * 255
        seed_masks.append(full_seed)

        bx0 = int(round(x0 + (mx - pad) * inv_scale))
        by0 = int(round(y0 + (my - pad) * inv_scale))
        bx1 = int(round(x0 + (mx + tw - pad) * inv_scale))
        by1 = int(round(y0 + (my + th - pad) * inv_scale))
        debug_matches.append(
            {
                "box": (bx0, by0, bx1, by1),
                "angle": tpl["angle"],
                "score": round(m["score"], 3),
            }
        )

    min_inst_area = max(25.0, 0.28 * proto.single_screw_area)
    instances = _partition_roi_by_seeds(
        roi_mask, seed_masks, x0, y0, min_inst_area, start_id, "screw_template_matching"
    )
    return instances, {"matches": debug_matches, "num_matches": len(matched_items)}


# =============================================================================
# Technique 3D: Contour Concavity Analysis
# =============================================================================

def _find_concave_points(contour: np.ndarray, min_depth: float) -> list[tuple[np.ndarray, float]]:
    """
    Detect concave pinch points along an outer contour using convexity defects
    and reflex-angle boundary vertices. Returns list of (point_xy, strength).
    """
    pts: list[tuple[np.ndarray, float]] = []
    if len(contour) < 6:
        return pts

    # 1. Convexity defects on simplified contour (avoids self-intersection errors)
    perim = cv2.arcLength(contour, True)
    eps = max(1.2, 0.003 * perim)
    approx = cv2.approxPolyDP(contour, eps, True)
    if len(approx) >= 4:
        hull_idx = cv2.convexHull(approx, returnPoints=False)
        if hull_idx is not None and len(hull_idx) >= 3:
            try:
                defects = cv2.convexityDefects(approx, hull_idx)
                if defects is not None:
                    for s, e, f, d_fixed in defects.reshape(-1, 4):
                        depth = d_fixed / 256.0
                        if depth >= min_depth:
                            pts.append((approx[f, 0].astype(np.float32), float(depth)))
            except cv2.error:
                pass

    # 2. Local inward-turning (reflex) vertices on polygon approximation
    n = len(approx)
    if n >= 5:
        poly = approx[:, 0, :].astype(np.float32)
        # Ensure consistent orientation via signed area
        signed_area = cv2.contourArea(approx, oriented=True)
        sign = 1.0 if signed_area >= 0 else -1.0
        for i in range(n):
            p_prev = poly[(i - 1) % n]
            p_curr = poly[i]
            p_next = poly[(i + 1) % n]
            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
            if n1 < 2.0 or n2 < 2.0:
                continue
            cross = sign * float(v1[0] * v2[1] - v1[1] * v2[0])
            cos_ang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            ang_deg = float(np.degrees(np.arccos(cos_ang)))
            # Reflex vertex (pointing into foreground) with sharp bend
            if cross < 0 and ang_deg <= 145.0:
                pts.append((p_curr, (180.0 - ang_deg) / 10.0))

    # Deduplicate nearby concave points (keep strongest)
    pts.sort(key=lambda t: t[1], reverse=True)
    dedup: list[tuple[np.ndarray, float]] = []
    merge_r = max(4.0, min_depth * 0.8)
    for p, s in pts:
        if all(float(np.linalg.norm(p - q)) >= merge_r for q, _ in dedup):
            dedup.append((p, s))
    return dedup


def _segment_inside_ratio(mask: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Fraction of interior sample points along segment (p1, p2) lying inside foreground mask."""
    dist = float(np.linalg.norm(p2 - p1))
    n_samples = max(5, int(round(dist)))
    ts = np.linspace(0.12, 0.88, n_samples)
    xs = np.clip(np.round(p1[0] + ts * (p2[0] - p1[0])).astype(np.int32), 0, mask.shape[1] - 1)
    ys = np.clip(np.round(p1[1] + ts * (p2[1] - p1[1])).astype(np.int32), 0, mask.shape[0] - 1)
    return float(np.mean(mask[ys, xs] > 0))


def _split_submask_concavity(
    sub_mask: np.ndarray,
    proto: ScenePrototype,
    cuts_out: list[tuple[np.ndarray, np.ndarray]],
    depth: int = 0,
) -> list[np.ndarray]:
    """Recursively cut a merged screw mask along geometrically validated concavity necks."""
    area = float(np.count_nonzero(sub_mask))
    if depth >= 16 or area <= 1.30 * proto.single_screw_area:
        return [sub_mask]

    cnts, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return [sub_mask]
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    solidity = float(cv2.contourArea(cnt)) / max(float(cv2.contourArea(hull)), 1e-6)
    if area <= 1.48 * proto.single_screw_area and solidity >= 0.56:
        return [sub_mask]

    min_depth = max(3.0, 0.30 * proto.single_screw_shaft_width)
    concave = _find_concave_points(cnt, min_depth)
    if not concave:
        return [sub_mask]

    cnt_pts = cnt[:, 0, :].astype(np.float32)
    max_cut_len = max(10.0, 2.35 * proto.single_screw_head_radius)
    min_cut_len = 2.0
    min_piece_area = 0.42 * proto.single_screw_area
    cut_thickness = max(2, int(round(max(sub_mask.shape) / 180.0)))

    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []

    # Candidate cuts (A): pairs of concave points
    for i in range(len(concave)):
        p1, s1 = concave[i]
        for j in range(i + 1, len(concave)):
            p2, s2 = concave[j]
            L = float(np.linalg.norm(p2 - p1))
            if not (min_cut_len <= L <= max_cut_len):
                continue
            if _segment_inside_ratio(sub_mask, p1, p2) < 0.80:
                continue
            # Prefer short pinch between two strong concavities
            score = (s1 + s2 + 8.0) / (L + 2.0)
            candidates.append((score, p1, p2))

    # Candidate cuts (B): single deep concave point to nearest opposite contour boundary
    step = max(1, len(cnt_pts) // 240)
    sampled_cnt = cnt_pts[::step]
    for p1, s1 in concave[:18]:
        dists = np.linalg.norm(sampled_cnt - p1, axis=1)
        # Filter to boundary points at reasonable cut distance and not adjacent along contour
        valid_idx = np.where((dists >= max(3.5, 0.25 * proto.single_screw_shaft_width)) & (dists <= max_cut_len))[0]
        if valid_idx.size == 0:
            continue
        # Test top 4 closest opposite boundary points
        order = valid_idx[np.argsort(dists[valid_idx])[:4]]
        for idx in order:
            p2 = sampled_cnt[idx]
            L = float(dists[idx])
            if _segment_inside_ratio(sub_mask, p1, p2) < 0.82:
                continue
            score = (s1 + 2.0) / (L + 2.5)
            candidates.append((score, p1, p2))

    if not candidates:
        return [sub_mask]

    candidates.sort(key=lambda t: t[0], reverse=True)

    # Greedily accept the highest-scoring cut that splits into two valid screw-sized parts
    for _, p1, p2 in candidates[:25]:
        trial = sub_mask.copy()
        # Slightly extend cut endpoints by 1px to guarantee clean topological separation
        d = p2 - p1
        dn = max(float(np.linalg.norm(d)), 1e-6)
        ep1 = np.round(p1 - (d / dn) * 1.5).astype(np.int32)
        ep2 = np.round(p2 + (d / dn) * 1.5).astype(np.int32)
        cv2.line(trial, (int(ep1[0]), int(ep1[1])), (int(ep2[0]), int(ep2[1])), 0, thickness=cut_thickness)

        n_cc, cc_lbl, cc_stats, _ = cv2.connectedComponentsWithStats(trial, connectivity=4)
        if n_cc <= 2:
            continue

        piece_areas = sorted(
            [(int(cc_stats[k, cv2.CC_STAT_AREA]), k) for k in range(1, n_cc)],
            reverse=True,
        )
        # Both largest pieces must be at least 42% of a single screw (never cuts off a tiny head!)
        if piece_areas[0][0] >= min_piece_area and piece_areas[1][0] >= min_piece_area:
            cuts_out.append((p1.copy(), p2.copy()))
            results: list[np.ndarray] = []
            for p_area, lbl in piece_areas:
                if p_area >= 0.25 * proto.single_screw_area:
                    piece = (cc_lbl == lbl).astype(np.uint8) * 255
                    results.extend(_split_submask_concavity(piece, proto, cuts_out, depth + 1))
            return results

    return [sub_mask]


def separate_contour_concavity(
    comp: ComponentFeatures,
    proto: ScenePrototype,
    start_id: int = 1,
) -> tuple[list[InstanceMask], dict]:
    """
    Technique 3D: Contour Concavity Analysis.
    Detects concave pinch points along merged screw boundaries and recursively applies
    geometrically validated neck cuts (rejecting cuts that would amputate a single screw head).
    """
    x0, y0, _, _ = comp.bbox
    cuts_local: list[tuple[np.ndarray, np.ndarray]] = []
    pieces = _split_submask_concavity(comp.roi_mask, proto, cuts_local, depth=0)

    instances: list[InstanceMask] = []
    next_id = start_id
    for p_mask in pieces:
        inst = _mask_to_instance(p_mask, x0, y0, next_id, "screw_contour_concavity")
        if inst is not None:
            instances.append(inst)
            next_id += 1

    cuts_global = [
        ((int(round(x0 + p1[0])), int(round(y0 + p1[1]))), (int(round(x0 + p2[0])), int(round(y0 + p2[1]))))
        for p1, p2 in cuts_local
    ]
    return instances, {"cuts": cuts_global, "num_cuts": len(cuts_global)}


# =============================================================================
# Technique 3E: Geometrically Constrained Watershed
# =============================================================================

def separate_constrained_watershed(
    comp: ComponentFeatures,
    proto: ScenePrototype,
    start_id: int = 1,
) -> tuple[list[InstanceMask], dict]:
    """
    Technique 3E: Geometrically Constrained Watershed.
    Extracts distance-transform peaks and merges peaks belonging to the same screw
    body using a geodesic valley-depth criterion (prevents shaft over-segmentation),
    then applies marker-controlled watershed on the inverted smoothed distance map.
    """
    x0, y0, bw, bh = comp.bbox
    roi_mask = comp.roi_mask
    roi_dist = comp.roi_dist

    max_d = float(roi_dist.max())
    if max_d <= 1.5:
        inst = _mask_to_instance(roi_mask, x0, y0, start_id, "screw_constrained_watershed")
        return ([inst] if inst else []), {"peaks": []}

    blur_k = max(3, int(round(0.45 * proto.single_screw_shaft_width)) | 1)
    blur_k = min(blur_k, 31)
    d_smooth = cv2.GaussianBlur(roi_dist, (blur_k, blur_k), 0)
    d_smooth[roi_mask == 0] = 0.0

    # Local maxima detection
    nms_k = max(5, int(round(0.85 * proto.single_screw_head_radius)) | 1)
    nms_k = min(nms_k, 61)
    dil = cv2.dilate(d_smooth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nms_k, nms_k)))
    min_peak_d = max(2.0, min(0.50 * proto.single_screw_head_radius, 0.48 * float(d_smooth.max())))
    pk_mask = (d_smooth == dil) & (d_smooth >= min_peak_d) & (roi_mask > 0)

    n_pk, pk_lbl, _, pk_cent = cv2.connectedComponentsWithStats(pk_mask.astype(np.uint8), connectivity=8)
    raw_peaks: list[tuple[float, float, float]] = []  # (dist_val, x, y)
    for k in range(1, n_pk):
        px, py = float(pk_cent[k][0]), float(pk_cent[k][1])
        dv = float(d_smooth[int(round(py)), int(round(px))])
        raw_peaks.append((dv, px, py))

    # Sort strongest head peaks first and merge same-screw shaft peaks via ridge/valley test
    raw_peaks.sort(key=lambda t: t[0], reverse=True)
    kept_peaks: list[tuple[float, float, float]] = []
    min_sep_close = max(4.0, 1.35 * proto.single_screw_head_radius)
    max_same_screw_len = 0.85 * proto.single_screw_length

    for dv, px, py in raw_peaks:
        merge = False
        for kdv, kx, ky in kept_peaks:
            dist_xy = float(np.hypot(px - kx, py - ky))
            if dist_xy < min_sep_close:
                merge = True
                break
            if dist_xy <= max_same_screw_len:
                # Sample distance transform along straight line connecting peaks
                n_s = max(6, int(round(dist_xy)))
                ts = np.linspace(0.0, 1.0, n_s)
                xs = np.clip(np.round(px + ts * (kx - px)).astype(np.int32), 0, bw - 1)
                ys = np.clip(np.round(py + ts * (ky - py)).astype(np.int32), 0, bh - 1)
                if np.all(roi_mask[ys, xs] > 0):
                    min_valley = float(d_smooth[ys, xs].min())
                    # No narrow pinch between peaks -> they lie along the same screw!
                    if min_valley >= 0.62 * min(dv, kdv):
                        merge = True
                        break
        if not merge:
            kept_peaks.append((dv, px, py))

    # Geometric count cap to prevent excessive splitting
    max_expected = max(2, int(np.ceil(comp.area / max(0.55 * proto.single_screw_area, 1.0))))
    kept_peaks = kept_peaks[:max_expected]

    seed_masks: list[np.ndarray] = []
    debug_peaks: list[tuple[int, int]] = []
    for dv, px, py in kept_peaks:
        sm = np.zeros_like(roi_mask)
        r_seed = max(2, int(round(0.35 * dv)))
        cv2.circle(sm, (int(round(px)), int(round(py))), r_seed, 255, thickness=cv2.FILLED)
        seed_masks.append(sm)
        debug_peaks.append((int(round(x0 + px)), int(round(y0 + py))))

    # Watershed on inverted smoothed distance transform (so dams snap to narrow necks)
    if len(seed_masks) <= 1:
        inst = _mask_to_instance(roi_mask, x0, y0, start_id, "screw_constrained_watershed")
        return ([inst] if inst else []), {"peaks": debug_peaks}

    markers = np.zeros((bh, bw), dtype=np.int32)
    markers[roi_mask == 0] = 1
    for idx, sm in enumerate(seed_masks, start=2):
        markers[(sm > 0) & (markers == 0)] = idx

    d_norm = cv2.normalize(d_smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    inv_d_bgr = cv2.cvtColor(255 - d_norm, cv2.COLOR_GRAY2BGR)
    cv2.watershed(inv_d_bgr, markers)

    dam = (markers == -1) & (roi_mask > 0)
    if np.any(dam):
        dil_m = cv2.dilate(np.where(markers >= 2, markers, 0).astype(np.uint16), np.ones((3, 3), np.uint8))
        markers[dam] = dil_m[dam].astype(np.int32)

    min_inst_area = max(25.0, 0.28 * proto.single_screw_area)
    instances: list[InstanceMask] = []
    next_id = start_id
    for idx in range(2, len(seed_masks) + 2):
        sub = ((markers == idx) & (roi_mask > 0)).astype(np.uint8) * 255
        if np.count_nonzero(sub) < min_inst_area:
            continue
        inst = _mask_to_instance(sub, x0, y0, next_id, "screw_constrained_watershed")
        if inst is not None:
            instances.append(inst)
            next_id += 1

    if not instances:
        inst = _mask_to_instance(roi_mask, x0, y0, start_id, "screw_constrained_watershed")
        return ([inst] if inst else []), {"peaks": debug_peaks}
    return instances, {"peaks": debug_peaks, "num_peaks": len(debug_peaks)}


# =============================================================================
# Final Combined Adaptive Screw Solver (Concavity + Template/Skeleton/Constrained)
# =============================================================================

def separate_hybrid_adaptive(
    comp: ComponentFeatures,
    proto: ScenePrototype,
    template_bank: list[dict],
    bank_scale: float,
    start_id: int = 1,
) -> tuple[list[InstanceMask], dict]:
    """
    Adaptive multi-stage screw separator for the final pipeline:
      1. First apply exact Contour Concavity neck cutting (ideal for touching screws).
      2. For any piece that remains merged (e.g. parallel flush or crossing overlaps),
         refine using Rotation-Invariant Template Matching (or Geometrically Constrained
         Watershed / Skeleton Graph if template match cannot split further).
    """
    x0, y0, _, _ = comp.bbox
    cuts_local: list[tuple[np.ndarray, np.ndarray]] = []
    pieces = _split_submask_concavity(comp.roi_mask, proto, cuts_local, depth=0)

    instances: list[InstanceMask] = []
    next_id = start_id
    used_methods: list[str] = []

    for p_mask in pieces:
        p_area = float(np.count_nonzero(p_mask))
        if p_area < max(25.0, 0.25 * proto.single_screw_area):
            continue

        # Check if piece is already cleanly isolated
        if p_area <= 1.52 * proto.single_screw_area:
            inst = _mask_to_instance(p_mask, x0, y0, next_id, "screw_hybrid_concavity")
            if inst is not None:
                instances.append(inst)
                next_id += 1
                used_methods.append("concavity")
            continue

        # Piece is STILL merged (e.g. X-crossing or flush parallel overlap)
        ys, xs = np.where(p_mask > 0)
        py0, py1 = int(ys.min()), int(ys.max()) + 1
        px0, px1 = int(xs.min()), int(xs.max()) + 1
        sub_roi = p_mask[py0:py1, px0:px1]
        sub_dist = cv2.distanceTransform(sub_roi, cv2.DIST_L2, 5)
        sub_comp = ComponentFeatures(
            comp_id=comp.comp_id,
            bbox=(x0 + px0, y0 + py0, px1 - px0, py1 - py0),
            outer_contour=comp.outer_contour,
            hole_contours=[],
            valid_hole_contours=[],
            area=p_area,
            filled_area=p_area,
            aspect_ratio=comp.aspect_ratio,
            major_length=max(px1 - px0, py1 - py0),
            minor_width=min(px1 - px0, py1 - py0),
            circularity=comp.circularity,
            solidity=comp.solidity,
            num_holes=0,
            max_dist=float(sub_dist.max()),
            mean_dist=float(sub_dist[sub_roi > 0].mean()),
            dist_peaks=2,
            skeleton_length=0.0,
            skeleton_endpoints=0,
            skeleton_junctions=0,
            mean_contrast=comp.mean_contrast,
            case_label="touching_screws",
            roi_mask=sub_roi,
            roi_dist=sub_dist,
        )

        # Try Rotation-Invariant Template Matching first on residual overlaps
        expected_k = max(2, int(round(p_area / max(proto.single_screw_area, 1.0))))
        tm_insts, _ = separate_template_matching(sub_comp, proto, template_bank, bank_scale, next_id)
        cw_insts, _ = separate_constrained_watershed(sub_comp, proto, next_id)
        sk_insts, _ = separate_skeleton_graph(sub_comp, proto, next_id)

        # Pick the candidate whose split count is closest to geometric area expectation (and >= 2)
        cands = [
            ("template", tm_insts),
            ("constrained_ws", cw_insts),
            ("skeleton", sk_insts),
        ]
        valid_cands = [(name, lst) for name, lst in cands if len(lst) >= 2]
        if valid_cands:
            best_name, best_lst = min(
                valid_cands,
                key=lambda item: (abs(len(item[1]) - expected_k), 0 if item[0] == "template" else 1),
            )
            for inst in best_lst:
                inst.instance_id = next_id
                inst.kind = f"screw_hybrid_{best_name}"
                instances.append(inst)
                next_id += 1
            used_methods.append(best_name)
        else:
            inst = _mask_to_instance(p_mask, x0, y0, next_id, "screw_hybrid_single")
            if inst is not None:
                instances.append(inst)
                next_id += 1
                used_methods.append("single")

    return instances, {"methods": used_methods, "num_concavity_cuts": len(cuts_local)}
