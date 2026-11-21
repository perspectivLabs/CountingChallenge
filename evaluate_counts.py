"""Evaluate image-count predictions against the ground-truth counts available."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _read_counts(
    path: Path, label: str, *, allow_missing_counts: bool = False
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not {"image", "count"}.issubset(frame.columns):
        raise ValueError(f"{label} CSV must contain image,count columns: {path}")
    frame = frame[["image", "count"]].copy()
    frame["image_key"] = frame["image"].astype(str).map(lambda value: Path(value).name)
    frame["count"] = pd.to_numeric(frame["count"], errors="raise")
    invalid = frame["count"].notna() & (
        (frame["count"] < 0) | (frame["count"] % 1 != 0)
    )
    if invalid.any() or (not allow_missing_counts and frame["count"].isna().any()):
        raise ValueError(f"{label} counts must be non-negative integers: {path}")
    if frame["image_key"].duplicated().any():
        duplicates = frame.loc[frame["image_key"].duplicated(False), "image_key"].tolist()
        raise ValueError(f"{label} contains duplicate image filenames: {duplicates}")
    return frame


def evaluate(predictions_path: Path, ground_truth_path: Path, output_path: Path) -> None:
    predictions = _read_counts(
        predictions_path, "Predictions", allow_missing_counts=True
    )
    predictions = predictions.rename(
        columns={"image": "prediction_image", "count": "predicted_count"}
    )
    predictions["prediction_status"] = pd.read_csv(predictions_path).get(
        "status", pd.Series(["ok"] * len(predictions))
    ).astype(str)
    predictions.loc[
        predictions["predicted_count"].isna(), "prediction_status"
    ] = "error"

    ground_truth = _read_counts(ground_truth_path, "Ground-truth")
    ground_truth = ground_truth.rename(
        columns={"image": "ground_truth_image", "count": "ground_truth_count"}
    )

    report = predictions.merge(
        ground_truth[["image_key", "ground_truth_image", "ground_truth_count"]],
        on="image_key",
        how="outer",
        validate="one_to_one",
    )
    report["image"] = report["prediction_image"].combine_first(
        report["ground_truth_image"]
    )
    report["prediction_status"] = report["prediction_status"].fillna("missing")
    missing_counts = report["predicted_count"].isna()
    missing_success = missing_counts & report["prediction_status"].eq("ok")
    report.loc[missing_success, "prediction_status"] = "error"
    report["ground_truth_status"] = np.where(
        report["ground_truth_count"].notna(), "verified", "unverified"
    )
    comparable = (
        report["ground_truth_status"].eq("verified")
        & report["prediction_status"].eq("ok")
        & report["predicted_count"].notna()
    )
    report["absolute_error"] = (
        report["predicted_count"] - report["ground_truth_count"]
    ).abs().where(comparable)
    report["signed_error"] = (
        report["predicted_count"] - report["ground_truth_count"]
    ).where(comparable)
    report["exact_match"] = pd.Series(pd.NA, index=report.index, dtype="boolean")
    verified = report["ground_truth_status"].eq("verified")
    report.loc[verified, "exact_match"] = False
    report.loc[comparable, "exact_match"] = report.loc[
        comparable, "absolute_error"
    ].eq(0).to_numpy()
    report["relative_count_accuracy"] = np.nan

    positive_truth = comparable & (report["ground_truth_count"] > 0)
    report.loc[positive_truth, "relative_count_accuracy"] = (
        1 - report.loc[positive_truth, "absolute_error"]
        / report.loc[positive_truth, "ground_truth_count"]
    )
    report["absolute_percentage_error"] = np.nan
    report.loc[positive_truth, "absolute_percentage_error"] = (
        100
        * report.loc[positive_truth, "absolute_error"]
        / report.loc[positive_truth, "ground_truth_count"]
    )
    zero_truth_exact = (
        comparable
        & report["ground_truth_count"].eq(0)
        & report["absolute_error"].eq(0)
    )
    report.loc[zero_truth_exact, "relative_count_accuracy"] = 1.0

    report = report[
        [
            "image",
            "predicted_count",
            "ground_truth_count",
            "absolute_error",
            "signed_error",
            "relative_count_accuracy",
            "absolute_percentage_error",
            "exact_match",
            "ground_truth_status",
            "prediction_status",
        ]
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    verified_rows = report[report["ground_truth_status"] == "verified"]
    scored_rows = verified_rows[verified_rows["absolute_error"].notna()]
    accuracy_values = scored_rows["relative_count_accuracy"].dropna()
    mape_values = scored_rows["absolute_percentage_error"].dropna()
    exact_values = verified_rows["exact_match"].fillna(False).astype(bool)
    exact_matches = int(exact_values.sum())
    exact_match_percentage = (
        exact_matches / len(verified_rows) * 100 if len(verified_rows) else None
    )
    print(f"Prediction rows   : {len(predictions)}")
    print(f"Ground-truth rows : {len(ground_truth)}")
    print(f"Verified images   : {len(verified_rows)}")
    print(f"Scored images     : {len(scored_rows)}")
    print(f"Unverified images : {(report['ground_truth_status'] == 'unverified').sum()}")
    print(
        "Mean relative count accuracy: "
        + (f"{accuracy_values.mean() * 100:.2f}%" if len(accuracy_values) else "N/A")
    )
    print(
        "Mean absolute percentage error: "
        + (f"{mape_values.mean():.2f}%" if len(mape_values) else "N/A")
    )
    print(
        "MAE              : "
        + (
            f"{scored_rows['absolute_error'].mean():.4f}"
            if len(scored_rows)
            else "N/A"
        )
    )
    print(f"Exact matches    : {exact_matches}/{len(verified_rows)}")
    print(
        "Exact-match rate : "
        + (f"{exact_match_percentage:.2f}%" if exact_match_percentage is not None else "N/A")
    )
    print(f"Per-image report : {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    evaluate(args.predictions, args.ground_truth, args.output)


if __name__ == "__main__":
    main()
