import json
import numpy as np
import pandas as pd

from itertools import combinations
from pathlib import Path

from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.splits import HOLDOUT_START
from earnings_surprise.outliers.compare_detectors import (
    COMPARISON_ASSIGNMENTS_PATH,
    COMPARISON_PATH,
)
from earnings_surprise.outliers.one_class_svm import (
    ONE_CLASS_ASSIGNMENTS_PATH,
    ONE_CLASS_RESULTS_PATH,
)
from earnings_surprise.outliers.prepare import (
    OUTLIER_COMPARISON_SAMPLE_SIZE,
    OUTLIER_RANDOM_STATE,
)

OUTLIER_LABEL = -1
INLIER_LABEL = 1

DETECTOR_COLUMNS = {
    "isolation_forest": "isolation_label",
    "local_outlier_factor": "lof_label",
    "one_class_svm": "svm_label",
}

ALL_DETECTORS_RESULTS_PATH = Path("results/outliers/three_detector_comparison.json")

ALL_DETECTORS_ASSIGNMENTS_PATH = Path("results/outliers/three_detector_assignments.csv")


def combine_saved_assignments(
    isolation_lof: pd.DataFrame,
    one_class_svm: pd.DataFrame,
) -> pd.DataFrame:
    """aligns anomaly assignments using their original row positions"""
    required_pairwise = {
        "source_row_position",
        "isolation_label",
        "isolation_score",
        "lof_label",
        "lof_score",
    }

    required_svm = {
        "source_row_position",
        "anomaly_label",
        "anomaly_score",
    }

    if not required_pairwise.issubset(
        isolation_lof.columns
    ) or not required_svm.issubset(one_class_svm.columns):
        raise ValueError("Missing anomaly assignment columns")

    if len(isolation_lof) == 0 or len(isolation_lof) != len(one_class_svm):
        raise ValueError("Assignment counts do not match")

    for assignments in (isolation_lof, one_class_svm):
        positions = assignments["source_row_position"]

        if (
            not pd.api.types.is_integer_dtype(positions)
            or positions.isna().any()
            or (positions < 0).any()
            or positions.duplicated().any()
        ):
            raise ValueError("Invalid source row positions")

    first_positions = isolation_lof["source_row_position"]
    second_positions = one_class_svm["source_row_position"]

    if set(first_positions) != set(second_positions):
        raise ValueError("Source row positions do not match")

    ordered_svm = one_class_svm.set_index("source_row_position").loc[
        first_positions.to_numpy()
    ]

    combined = isolation_lof[
        [
            "source_row_position",
            "isolation_label",
            "isolation_score",
            "lof_label",
            "lof_score",
        ]
    ].copy()

    combined["svm_label"] = ordered_svm["anomaly_label"].to_numpy()

    combined["svm_score"] = ordered_svm["anomaly_score"].to_numpy()

    for column in DETECTOR_COLUMNS.values():
        if not np.isin(
            combined[column].to_numpy(),
            [OUTLIER_LABEL, INLIER_LABEL],
        ).all():
            raise ValueError("Invalid anomaly labels")

    for column in ("isolation_score", "lof_score", "svm_score"):
        if not np.isfinite(combined[column].to_numpy(dtype=float)).all():
            raise ValueError("Anomaly scores must be finite")

    outlier_flags = np.column_stack(
        [
            combined[column].to_numpy() == OUTLIER_LABEL
            for column in DETECTOR_COLUMNS.values()
        ]
    )

    combined["detector_count"] = outlier_flags.sum(axis=1)

    combined["agreement_group"] = np.select(
        [
            combined["detector_count"] == 3,
            combined["detector_count"] == 2,
            combined["detector_count"] == 1,
        ],
        ["all_three", "exactly_two", "exactly_one"],
        default="none",
    )

    return combined


def summarize_three_detector_agreement(
    assignments: pd.DataFrame,
) -> dict:
    """summarizes individual, pairwise, three-way outlier overlap"""
    if assignments.empty:
        raise ValueError("Anomaly assignments cannot be empty")

    missing_columns = set(DETECTOR_COLUMNS.values()) - set(assignments.columns)

    if missing_columns:
        raise ValueError("Missing detector label columns")

    masks = {}

    for detector, column in DETECTOR_COLUMNS.items():
        labels = assignments[column].to_numpy()

        if not np.isin(
            labels,
            [OUTLIER_LABEL, INLIER_LABEL],
        ).all():
            raise ValueError("Invalid anomaly labels")

        masks[detector] = labels == OUTLIER_LABEL

    sample_count = len(assignments)

    detector_counts = {name: int(mask.sum()) for name, mask in masks.items()}

    detection_counts = np.column_stack(list(masks.values())).sum(axis=1)

    pairwise = {}

    for first, second in combinations(DETECTOR_COLUMNS, 2):
        first_mask = masks[first]
        second_mask = masks[second]

        intersection = int(np.sum(first_mask & second_mask))

        union = int(np.sum(first_mask | second_mask))

        pairwise[f"{first}_vs_{second}"] = {
            "intersection": intersection,
            "union": union,
            "jaccard_similarity": (intersection / union if union else None),
            "expected_random_overlap": (
                detector_counts[first] * detector_counts[second] / sample_count
            ),
        }

    all_three = int(np.sum(detection_counts == 3))
    flagged_by_any = int(np.sum(detection_counts > 0))

    return {
        "samples": sample_count,
        "detector_outliers": detector_counts,
        "agreement_counts": {
            "none": int(np.sum(detection_counts == 0)),
            "exactly_one": int(np.sum(detection_counts == 1)),
            "exactly_two": int(np.sum(detection_counts == 2)),
            "all_three": all_three,
            "flagged_by_any": flagged_by_any,
        },
        "pairwise_agreement": pairwise,
        "three_way_jaccard_similarity": (
            all_three / flagged_by_any if flagged_by_any else None
        ),
    }


def main() -> None:
    """compares three saved detectors on identical development rows"""
    isolation_lof = pd.read_csv(COMPARISON_ASSIGNMENTS_PATH)
    one_class_svm = pd.read_csv(ONE_CLASS_ASSIGNMENTS_PATH)

    with COMPARISON_PATH.open(encoding="utf-8") as file:
        isolation_lof_report = json.load(file)

    with ONE_CLASS_RESULTS_PATH.open(encoding="utf-8") as file:
        svm_report = json.load(file)

    assignments = combine_saved_assignments(
        isolation_lof,
        one_class_svm,
    )

    sample_count = len(assignments)

    if (
        sample_count != OUTLIER_COMPARISON_SAMPLE_SIZE
        or isolation_lof_report["sample_count"] != sample_count
        or svm_report["samples"] != sample_count
        or isolation_lof_report["feature_count"] != svm_report["feature_count"]
        or isolation_lof_report["random_state"] != OUTLIER_RANDOM_STATE
        or svm_report["sampling"]["random_state"] != OUTLIER_RANDOM_STATE
        or isolation_lof_report["holdout_used"] is not False
        or svm_report["sampling"]["holdout_used"] is not False
    ):
        raise ValueError("Detector experiment metadata does not match")

    samples = pd.read_parquet(
        PREDICTION_WINDOWS_PATH,
        columns=["target_date_pub"],
    )

    positions = assignments["source_row_position"].to_numpy()

    if np.any(positions >= len(samples)):
        raise ValueError("Source row positions exceed sample count")

    selected_dates = pd.to_datetime(samples.iloc[positions]["target_date_pub"])

    if selected_dates.isna().any() or (selected_dates >= HOLDOUT_START).any():
        raise ValueError("Holdout observations entered comparison")

    agreement = summarize_three_detector_agreement(assignments)

    expected_counts = {
        "isolation_forest": isolation_lof_report["isolation_forest"]["outlier_count"],
        "local_outlier_factor": isolation_lof_report["local_outlier_factor"][
            "outlier_count"
        ],
        "one_class_svm": svm_report["outlier_count"],
    }

    if agreement["detector_outliers"] != expected_counts:
        raise ValueError("Outlier counts differ from saved experiments")

    previous_overlap = isolation_lof_report["agreement"]["both_outliers"]

    current_overlap = agreement["pairwise_agreement"][
        "isolation_forest_vs_local_outlier_factor"
    ]["intersection"]

    if current_overlap != previous_overlap:
        raise ValueError("Isolation Forest and LOF overlap changed")

    report = {
        "experiment": "three_detector_outlier_comparison",
        "sample_count": sample_count,
        "feature_count": svm_report["feature_count"],
        "random_state": OUTLIER_RANDOM_STATE,
        "holdout_used": False,
        **agreement,
    }

    ALL_DETECTORS_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ALL_DETECTORS_RESULTS_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    assignments.to_csv(
        ALL_DETECTORS_ASSIGNMENTS_PATH,
        index=False,
    )

    print("Three-Algorithm Outlier Comparison")
    print("----------------------------------")

    print(f"Samples: {sample_count:,}")

    print("\nOutliers by detector:")

    for detector, count in agreement["detector_outliers"].items():
        print(f"  {detector}: {count}")

    print("\nAgreement groups:")

    for group, count in agreement["agreement_counts"].items():
        print(f"  {group}: {count}")

    print("\nPairwise Jaccard similarities:")

    for pair, result in agreement["pairwise_agreement"].items():
        similarity = result["jaccard_similarity"]

        print(
            f"  {pair}: {similarity:.4f}"
            if similarity is not None
            else f"  {pair}: undefined"
        )

    print(
        "\nThree-way Jaccard similarity: "
        f"{agreement['three_way_jaccard_similarity']:.4f}"
    )

    print(f"\nResults saved to: {ALL_DETECTORS_RESULTS_PATH}")
    print(f"Assignments saved to: {ALL_DETECTORS_ASSIGNMENTS_PATH}")


if __name__ == "__main__":
    main()
