from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from adaptive_pipeline.features import ComponentFeatures, extract_and_classify_components
from adaptive_pipeline.illumination import normalize_illumination_and_segment
from adaptive_pipeline.nuts import InstanceMask, separate_touching_nuts
from adaptive_pipeline.screws import (
    build_template_bank,
    separate_constrained_watershed,
    separate_contour_concavity,
    separate_hough_lines,
    separate_hybrid_adaptive,
    separate_skeleton_graph,
    separate_template_matching,
)
from adaptive_pipeline.visualization import (
    render_classification_overlay,
    render_instances_overlay,
    save_isolated_masks,
    save_png,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _isolated_component_to_instance(comp: ComponentFeatures, inst_id: int) -> InstanceMask:
    x0, y0, bw, bh = comp.bbox
    ys, xs = np.where(comp.roi_mask > 0)
    cx = int(round(x0 + xs.mean())) if xs.size else x0 + bw // 2
    cy = int(round(y0 + ys.mean())) if ys.size else y0 + bh // 2
    return InstanceMask(
        instance_id=inst_id,
        bbox=comp.bbox,
        mask=comp.roi_mask.copy(),
        center=(cx, cy),
        kind=comp.case_label,
        debug_info={},
    )


def process_single_image(
    image_path: Path,
    out_root: Path,
    feature_rows: list[dict],
) -> dict:
    t0 = time.perf_counter()
    stem = image_path.stem
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    h, w = img_bgr.shape[:2]

    # -------------------------------------------------------------------------
    # Stage 1: Illumination Normalization & Foreground Mask Generation
    # -------------------------------------------------------------------------
    illum = normalize_illumination_and_segment(img_bgr)
    save_png(out_root / "01_illumination_mask" / "normalized" / f"{stem}.png", illum.normalized)
    save_png(out_root / "01_illumination_mask" / "masks" / f"{stem}.png", illum.foreground_mask)

    # -------------------------------------------------------------------------
    # Stage 2: Connected Region Geometric Features & 3-Case Classification
    # -------------------------------------------------------------------------
    components, proto = extract_and_classify_components(illum.foreground_mask, illum.contrast)
    cls_overlay = render_classification_overlay(img_bgr, components, proto.scene_type)
    save_png(out_root / "02_geometric_classification" / "classification_overlays" / f"{stem}.png", cls_overlay)

    for comp in components:
        if comp.case_label == "noise":
            continue
        feature_rows.append(
            {
                "image": image_path.name,
                "scene_type": proto.scene_type,
                "comp_id": comp.comp_id,
                "case_label": comp.case_label,
                "area": round(comp.area, 1),
                "filled_area": round(comp.filled_area, 1),
                "aspect_ratio": round(comp.aspect_ratio, 3),
                "circularity": round(comp.circularity, 3),
                "solidity": round(comp.solidity, 3),
                "num_holes": comp.num_holes,
                "max_dist": round(comp.max_dist, 2),
                "dist_peaks": comp.dist_peaks,
                "skeleton_length": round(comp.skeleton_length, 1),
                "skeleton_endpoints": comp.skeleton_endpoints,
                "skeleton_junctions": comp.skeleton_junctions,
                "estimated_count": comp.estimated_count,
            }
        )

    isolated_comps = [c for c in components if c.case_label in ("isolated_screw", "isolated_nut")]
    touching_nut_comps = [c for c in components if c.case_label == "touching_nuts"]
    touching_screw_comps = [c for c in components if c.case_label == "touching_screws"]

    # -------------------------------------------------------------------------
    # Case 1: Isolated Objects (Direct Connected Component Masks)
    # -------------------------------------------------------------------------
    isolated_instances = [
        _isolated_component_to_instance(c, idx) for idx, c in enumerate(isolated_comps, start=1)
    ]
    iso_overlay = render_instances_overlay(
        img_bgr,
        isolated_instances,
        title_prefix="Case 1 Isolated",
        subtitle=f"Scene: {proto.scene_type} | Direct CCA Masks",
    )
    save_png(out_root / "03_case1_isolated" / f"{stem}.png", iso_overlay)

    # -------------------------------------------------------------------------
    # Case 2: Touching Nuts (Validated Hole-Marker Watershed)
    # -------------------------------------------------------------------------
    nut_ws_instances: list[InstanceMask] = []
    nut_debug_list: list[dict] = []
    next_id = len(isolated_instances) + 1
    for comp in touching_nut_comps:
        insts, holes = separate_touching_nuts(comp, proto, illum.normalized, start_id=next_id)
        nut_ws_instances.extend(insts)
        nut_debug_list.append({"holes": holes})
        next_id += len(insts)

    base_non_screw_cluster = isolated_instances + nut_ws_instances
    if proto.scene_type == "nuts" or touching_nut_comps:
        nut_overlay = render_instances_overlay(
            img_bgr,
            base_non_screw_cluster,
            title_prefix="Case 2 Nuts WS",
            subtitle=f"Isolated: {len(isolated_instances)} + Separated Nuts: {len(nut_ws_instances)}",
            debug_overlays=nut_debug_list,
        )
        save_png(out_root / "04_case2_touching_nuts" / f"{stem}.png", nut_overlay)

    # -------------------------------------------------------------------------
    # Case 3: Touching & Overlapping Screws (All 5 Techniques Evaluated Separately)
    # -------------------------------------------------------------------------
    template_bank, bank_scale = (
        build_template_bank(proto) if touching_screw_comps else ([], 1.0)
    )

    techniques = {
        "skeleton_graph": "Skeleton Graph Analysis",
        "hough_lines": "Hough Line Detection",
        "template_matching": "Rotation-Invariant Template Matching",
        "contour_concavity": "Contour Concavity Analysis",
        "constrained_watershed": "Geometrically Constrained Watershed",
    }

    tech_counts: dict[str, int] = {}

    for tech_key, tech_title in techniques.items():
        tech_screw_insts: list[InstanceMask] = []
        tech_debugs: list[dict] = []
        cur_id = len(base_non_screw_cluster) + 1

        for comp in touching_screw_comps:
            if tech_key == "skeleton_graph":
                insts, dbg = separate_skeleton_graph(comp, proto, cur_id)
            elif tech_key == "hough_lines":
                insts, dbg = separate_hough_lines(comp, proto, cur_id)
            elif tech_key == "template_matching":
                insts, dbg = separate_template_matching(comp, proto, template_bank, bank_scale, cur_id)
            elif tech_key == "contour_concavity":
                insts, dbg = separate_contour_concavity(comp, proto, cur_id)
            elif tech_key == "constrained_watershed":
                insts, dbg = separate_constrained_watershed(comp, proto, cur_id)
            else:
                insts, dbg = [], {}

            tech_screw_insts.extend(insts)
            tech_debugs.append(dbg)
            cur_id += len(insts)

        full_tech_insts = base_non_screw_cluster + tech_screw_insts
        tech_counts[tech_key] = len(full_tech_insts)

        # Always save per-technique overlay for screw scenes (and for all images so folders stay complete)
        tech_overlay = render_instances_overlay(
            img_bgr,
            full_tech_insts,
            title_prefix="Count",
            subtitle=f"{tech_title} | Clusters: {len(touching_screw_comps)}",
            debug_overlays=tech_debugs + nut_debug_list,
        )
        save_png(out_root / "05_case3_screw_techniques" / tech_key / f"{stem}.png", tech_overlay)

    # -------------------------------------------------------------------------
    # Final Combined Adaptive Pipeline + Isolated Instance Mask Export
    # -------------------------------------------------------------------------
    final_screw_insts: list[InstanceMask] = []
    cur_id = len(base_non_screw_cluster) + 1
    for comp in touching_screw_comps:
        insts, _ = separate_hybrid_adaptive(comp, proto, template_bank, bank_scale, cur_id)
        final_screw_insts.extend(insts)
        cur_id += len(insts)

    final_instances = base_non_screw_cluster + final_screw_insts
    final_overlay = render_instances_overlay(
        img_bgr,
        final_instances,
        title_prefix="Count",
        subtitle=f"Final Adaptive ({proto.scene_type}) | Iso:{len(isolated_instances)} NutWS:{len(nut_ws_instances)} ScrewSep:{len(final_screw_insts)}",
    )
    save_png(out_root / "06_final_adaptive" / "overlays" / f"{stem}.png", final_overlay)
    save_isolated_masks(
        (h, w),
        final_instances,
        out_root / "06_final_adaptive" / "label_masks" / f"{stem}_labels.png",
        out_root / "06_final_adaptive" / "isolated_masks" / stem,
    )

    elapsed = time.perf_counter() - t0
    row = {
        "image": image_path.name,
        "scene_type": proto.scene_type,
        "case1_isolated": len(isolated_instances),
        "case2_touching_nut_clusters": len(touching_nut_comps),
        "case2_separated_nuts": len(nut_ws_instances),
        "case3_touching_screw_clusters": len(touching_screw_comps),
        "skeleton_graph": tech_counts["skeleton_graph"],
        "hough_lines": tech_counts["hough_lines"],
        "template_matching": tech_counts["template_matching"],
        "contour_concavity": tech_counts["contour_concavity"],
        "constrained_watershed": tech_counts["constrained_watershed"],
        "final_adaptive": len(final_instances),
        "elapsed_sec": round(elapsed, 2),
    }
    print(
        f"[{image_path.name:20s}] {proto.scene_type:6s} | "
        f"Iso={len(isolated_instances):3d} NutC={len(touching_nut_comps):2d} ScrC={len(touching_screw_comps):2d} | "
        f"Skel={tech_counts['skeleton_graph']:3d} Hough={tech_counts['hough_lines']:3d} "
        f"Tpl={tech_counts['template_matching']:3d} Conc={tech_counts['contour_concavity']:3d} "
        f"CWS={tech_counts['constrained_watershed']:3d} | FINAL={len(final_instances):3d} ({elapsed:.2f}s)"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive Geometric-Based Instance Segmentation")
    parser.add_argument("--images", type=Path, default=None, help="Path to input images directory")
    parser.add_argument("--output", type=Path, default=None, help="Path to output results directory")
    parser.add_argument("--only", type=str, nargs="*", default=None, help="Optional image stems to process")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    images_dir = args.images or (script_dir.parent / "images")
    out_root = args.output or (script_dir / "adaptive_results")
    out_root.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if args.only:
        only_set = set(args.only)
        image_paths = [p for p in image_paths if p.stem in only_set or p.name in only_set]

    print(f"Processing {len(image_paths)} images from {images_dir} -> {out_root}\n")

    summary_rows: list[dict] = []
    feature_rows: list[dict] = []

    t_total = time.perf_counter()
    for img_path in image_paths:
        row = process_single_image(img_path, out_root, feature_rows)
        summary_rows.append(row)

    # Save summary_counts.csv
    if summary_rows:
        summary_csv = out_root / "summary_counts.csv"
        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    # Save geometric_features.csv
    if feature_rows:
        feat_csv = out_root / "02_geometric_classification" / "geometric_features.csv"
        feat_csv.parent.mkdir(parents=True, exist_ok=True)
        with feat_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(feature_rows[0].keys()))
            writer.writeheader()
            writer.writerows(feature_rows)

    print(f"\nDone in {time.perf_counter() - t_total:.2f}s. Saved summary to {out_root / 'summary_counts.csv'}")


if __name__ == "__main__":
    main()
