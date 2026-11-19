from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .features import ComponentFeatures, ScenePrototype, contour_circularity


@dataclass
class InstanceMask:
    instance_id: int
    bbox: tuple[int, int, int, int]  # (x, y, w, h) in full image coordinates
    mask: np.ndarray  # uint8 (0 or 255) tight ROI mask matching bbox
    center: tuple[int, int]  # (cx, cy) in full image coordinates
    kind: str  # "isolated_nut" | "isolated_screw" | "nut_watershed" | "screw_<technique>"
    debug_info: dict


def validate_nut_holes(
    comp: ComponentFeatures,
    proto: ScenePrototype,
) -> list[dict]:
    """
    Steps 1 & 2 of Touching Nuts algorithm:
    Detect candidate internal holes from contour hierarchy and validate each hole using:
      - Area consistency around median hole area
      - Circularity & solidity (rejects sharp concave interstitial gaps between touching nuts)
      - Surrounding foreground ring coverage (verifies hole is enclosed by nut metal)
      - Physical non-overlap center-to-center distance (rejects tiny triangular gaps)
    """
    x0, y0, bw, bh = comp.bbox
    roi_mask = comp.roi_mask
    med_area = max(proto.median_hole_area, 12.0)
    med_r = max(proto.median_hole_radius, 3.0)

    candidates: list[dict] = []
    offset = np.array([[[x0, y0]]], dtype=np.int32)

    for hc in comp.hole_contours:
        ha = float(cv2.contourArea(hc))
        if not (0.22 * med_area <= ha <= 3.2 * med_area):
            continue

        circ = contour_circularity(hc)
        hull = cv2.convexHull(hc)
        hull_a = max(float(cv2.contourArea(hull)), 1e-6)
        sol = ha / hull_a
        if circ < 0.52 or sol < 0.78:
            continue

        shifted = hc - offset
        (cx_l, cy_l), r = cv2.minEnclosingCircle(shifted)
        if r < 0.45 * med_r or r > 2.2 * med_r:
            continue

        # Check surrounding annular ring [1.25*r, 2.2*r] is predominantly foreground
        ring_mask = np.zeros_like(roi_mask)
        cv2.circle(
            ring_mask,
            (int(round(cx_l)), int(round(cy_l))),
            max(3, int(round(r * 2.15))),
            255,
            thickness=cv2.FILLED,
        )
        cv2.circle(
            ring_mask,
            (int(round(cx_l)), int(round(cy_l))),
            max(1, int(round(r * 1.20))),
            0,
            thickness=cv2.FILLED,
        )
        ring_total = int(np.count_nonzero(ring_mask))
        if ring_total == 0:
            continue
        ring_fg = int(np.count_nonzero(cv2.bitwise_and(ring_mask, roi_mask)))
        fg_ratio = ring_fg / ring_total
        if fg_ratio < 0.55:
            continue

        # Score favors high circularity, high foreground ring enclosure, and area near median
        area_score = 1.0 - min(1.0, abs(ha - med_area) / (2.0 * med_area))
        score = 0.45 * circ + 0.35 * fg_ratio + 0.20 * area_score

        candidates.append(
            {
                "contour_local": shifted,
                "center_local": (float(cx_l), float(cy_l)),
                "center_global": (int(round(x0 + cx_l)), int(round(y0 + cy_l))),
                "radius": float(r),
                "area": ha,
                "circularity": circ,
                "fg_ratio": fg_ratio,
                "score": score,
            }
        )

    # Physical non-overlap suppression: real nut centers are separated by >= ~2.45 * med_r
    # This cleanly suppresses any trapped interstitial gap between 3 touching nuts
    min_center_dist = 2.45 * med_r
    candidates.sort(key=lambda d: d["score"], reverse=True)
    validated: list[dict] = []
    for cand in candidates:
        cx, cy = cand["center_local"]
        too_close = False
        for kept in validated:
            kx, ky = kept["center_local"]
            if np.hypot(cx - kx, cy - ky) < min_center_dist:
                too_close = True
                break
        if not too_close:
            validated.append(cand)

    return validated


def separate_touching_nuts(
    comp: ComponentFeatures,
    proto: ScenePrototype,
    normalized_gray: np.ndarray,
    start_id: int = 1,
) -> tuple[list[InstanceMask], list[dict]]:
    """
    Case 2: Touching nuts separation via validated internal-hole marker-controlled watershed.
      1. Detect candidate internal holes via contour hierarchy.
      2. Validate holes using size, circularity, ring coverage, and non-overlap geometry.
      3. Generate markers from validated holes on the hole-filled cluster.
      4. Apply marker-controlled watershed on distance-transform + intensity seam topography.
      5. Preserve internal holes by intersecting each watershed basin with the original mask.
    """
    x0, y0, bw, bh = comp.bbox
    roi_mask = comp.roi_mask
    validated_holes = validate_nut_holes(comp, proto)

    if len(validated_holes) <= 1:
        # Fallback: single nut (or unresolvable cluster) -> keep directly with holes preserved
        ys, xs = np.where(roi_mask > 0)
        cx = int(round(x0 + xs.mean())) if xs.size else x0 + bw // 2
        cy = int(round(y0 + ys.mean())) if ys.size else y0 + bh // 2
        inst = InstanceMask(
            instance_id=start_id,
            bbox=comp.bbox,
            mask=roi_mask.copy(),
            center=(cx, cy),
            kind="nut_watershed",
            debug_info={"holes": validated_holes},
        )
        return [inst], validated_holes

    # Step 3: Fill the cluster outer contour so watershed floods cleanly across nut bodies
    filled_cluster = np.zeros_like(roi_mask)
    offset = np.array([[[x0, y0]]], dtype=comp.outer_contour.dtype)
    cv2.drawContours(filled_cluster, [comp.outer_contour - offset], -1, 255, thickness=cv2.FILLED)

    # Also mask out any LARGE non-nut background cavities inside the cluster (e.g. big empty
    # lakes enclosed by a ring of touching nuts) so watershed stays strictly on nut metal + nut holes
    valid_hole_mask = np.zeros_like(roi_mask)
    for h_info in validated_holes:
        cv2.drawContours(valid_hole_mask, [h_info["contour_local"]], -1, 255, thickness=cv2.FILLED)
    # Valid nut domain = actual foreground metal UNION validated nut holes
    nut_domain = cv2.bitwise_or(roi_mask, valid_hole_mask)

    # Build watershed markers: 1 = exterior background, 2..N+1 = validated nut hole seeds
    markers = np.zeros(roi_mask.shape, dtype=np.int32)
    markers[nut_domain == 0] = 1

    for idx, h_info in enumerate(validated_holes, start=2):
        cx_l, cy_l = h_info["center_local"]
        seed_r = max(2, int(round(h_info["radius"] * 0.85)))
        cv2.circle(markers, (int(round(cx_l)), int(round(cy_l))), seed_r, idx, thickness=cv2.FILLED)

    # Step 4: Marker-controlled watershed on strictly convex quadratic distance-from-seeds surface.
    # Because cv2.watershed floods by step difference |I(u) - I(v)|, a quadratic bowl
    # I(r) = (r / max_r)^2 has step gradient proportional to r, guaranteeing all nut hole
    # markers expand symmetrically and meet precisely at the shared contact seams.
    seed_binary = (markers >= 2).astype(np.uint8)
    dist_from_seeds = cv2.distanceTransform(1 - seed_binary, cv2.DIST_L2, 5)
    max_r = max(10.0, 3.2 * proto.median_hole_radius)
    quad_surface = np.clip((dist_from_seeds / max_r) ** 2 * 255.0, 0, 255).astype(np.uint8)
    surface_bgr = cv2.cvtColor(quad_surface, cv2.COLOR_GRAY2BGR)
    cv2.watershed(surface_bgr, markers)

    # Assign 1px watershed boundary pixels inside roi_mask to nearest valid nut label
    dam = (markers == -1) & (roi_mask > 0)
    if np.any(dam):
        dil = cv2.dilate(np.where(markers >= 2, markers, 0).astype(np.uint16), np.ones((3, 3), np.uint8))
        markers[dam] = dil[dam].astype(np.int32)

    # Step 5: Extract individual instance masks while PRESERVING internal holes
    instances: list[InstanceMask] = []
    next_id = start_id
    min_inst_area = max(15.0, 0.15 * proto.single_nut_filled_area)

    for idx, h_info in enumerate(validated_holes, start=2):
        # Intersect basin with original roi_mask (holes are 0 in roi_mask -> preserved!)
        inst_roi = ((markers == idx) & (roi_mask > 0)).astype(np.uint8) * 255
        if np.count_nonzero(inst_roi) < min_inst_area:
            continue

        ys, xs = np.where(inst_roi > 0)
        iy0, iy1 = int(ys.min()), int(ys.max()) + 1
        ix0, ix1 = int(xs.min()), int(xs.max()) + 1
        tight_mask = inst_roi[iy0:iy1, ix0:ix1]

        instances.append(
            InstanceMask(
                instance_id=next_id,
                bbox=(x0 + ix0, y0 + iy0, ix1 - ix0, iy1 - iy0),
                mask=tight_mask,
                center=h_info["center_global"],
                kind="nut_watershed",
                debug_info={"hole_center": h_info["center_global"], "hole_radius": h_info["radius"]},
            )
        )
        next_id += 1

    return instances, validated_holes
