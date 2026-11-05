import json
import numpy as np
import pandas as pd

from pathlib import Path
from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.splits import HOLDOUT_START
from earnings_surprise.features import FEATURE_MATRIX_PATH
from earnings_surprise.outliers.isolation_forest import (
    run_isolation_forest,
)
from earnings_surprise.outliers.local_outlier_factor import (
    LOF_ASSIGNMENTS_PATH,
    LOF_RESULTS_PATH,
    run_local_outlier_factor,
)
from earnings_surprise.outliers.prepare import (
    OUTLIER_COMPARISON_SAMPLE_SIZE,
    OUTLIER_RANDOM_STATE,
    prepare_outlier_data,
)

OUTLIER_LABEL = -1
SCORE_TOLERANCE = 1e-10

COMPARISON_PATH = Path("results/outliers/isolation_lof_comparison.json")

COMPARISON_ASSIGNMENTS_PATH = Path("results/outliers/isolation_lof_assignments.csv")


def summarize_detector_agreement(
    isolation_labels: np.ndarray,
    lof_labels: np.ndarray,
) -> dict:
    """measures overlap between two sets of outlier assignments"""
    isolation_labels = np.asarray(isolation_labels)
    lof_labels = np.asarray(lof_labels)

    if (
        isolation_labels.ndim != 1
        or lof_labels.ndim != 1
        or len(isolation_labels) == 0
        or len(isolation_labels) != len(lof_labels)
    ):
        raise ValueError("Outlier label arrays must have matching lengths")

    if (
        not np.isin(isolation_labels, [-1, 1]).all()
        or not np.isin(lof_labels, [-1, 1]).all()
    ):
        raise ValueError("Invalid outlier labels")

    isolation_outliers = isolation_labels == OUTLIER_LABEL
    lof_outliers = lof_labels == OUTLIER_LABEL

    both = int(np.sum(isolation_outliers & lof_outliers))
    isolation_only = int(np.sum(isolation_outliers & ~lof_outliers))
    lof_only = int(np.sum(~isolation_outliers & lof_outliers))
    neither = int(np.sum(~isolation_outliers & ~lof_outliers))

    isolation_count = int(isolation_outliers.sum())
    lof_count = int(lof_outliers.sum())
    union_count = both + isolation_only + lof_only

    jaccard = both / union_count if union_count > 0 else None

    expected_random_overlap = isolation_count * lof_count / len(isolation_labels)

    return {
        "samples": len(isolation_labels),
        "isolation_outliers": isolation_count,
        "lof_outliers": lof_count,
        "both_outliers": both,
        "isolation_only": isolation_only,
        "lof_only": lof_only,
        "neither": neither,
        "union_outliers": union_count,
        "jaccard_similarity": jaccard,
        "expected_overlap_under_random_selection": (expected_random_overlap),
    }


def verify_saved_lof_assignments(
    saved: pd.DataFrame,
    positions: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
) -> None:
    """verifies that LOF reproduces the saved row-level results"""
    required_columns = {
        "source_row_position",
        "anomaly_label",
        "anomaly_score",
    }

    if not required_columns.issubset(saved.columns):
        raise ValueError("Missing saved LOF assignment columns")

    if not np.array_equal(
        saved["source_row_position"].to_numpy(),
        positions,
    ):
        raise ValueError("Saved LOF sample positions do not match")

    if not np.array_equal(
        saved["anomaly_label"].to_numpy(),
        labels,
    ):
        raise ValueError("Saved LOF labels changed")

    if not np.allclose(
        saved["anomaly_score"].to_numpy(dtype=float),
        scores,
        rtol=0,
        atol=SCORE_TOLERANCE,
    ):
        raise ValueError("Saved LOF scores changed")


def main() -> None:
    """compares Isolation Forest and LOF on identical data."""
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    data, positions, _ = prepare_outlier_data(
        features,
        samples,
        sample_size=OUTLIER_COMPARISON_SAMPLE_SIZE,
    )

    selected_dates = pd.to_datetime(samples.iloc[positions]["target_date_pub"])

    if (selected_dates >= HOLDOUT_START).any():
        raise ValueError("Holdout observations entered comparison")

    with LOF_RESULTS_PATH.open(encoding="utf-8") as file:
        saved_lof_summary = json.load(file)

    if (
        saved_lof_summary["sampling"]["holdout_used"] is not False
        or saved_lof_summary["samples"] != len(data)
        or saved_lof_summary["feature_count"] != data.shape[1]
    ):
        raise ValueError("Saved LOF experiment does not match sample")

    isolation_labels, isolation_scores, isolation_summary = run_isolation_forest(data)

    lof_labels, lof_scores, lof_summary = run_local_outlier_factor(data)

    saved_assignments = pd.read_csv(LOF_ASSIGNMENTS_PATH)

    verify_saved_lof_assignments(
        saved_assignments,
        positions,
        lof_labels,
        lof_scores,
    )

    agreement = summarize_detector_agreement(
        isolation_labels,
        lof_labels,
    )

    isolation_outliers = isolation_labels == OUTLIER_LABEL
    lof_outliers = lof_labels == OUTLIER_LABEL

    categories = np.select(
        [
            isolation_outliers & lof_outliers,
            isolation_outliers & ~lof_outliers,
            ~isolation_outliers & lof_outliers,
        ],
        [
            "both",
            "isolation_only",
            "lof_only",
        ],
        default="neither",
    )

    assignments = pd.DataFrame(
        {
            "source_row_position": positions,
            "isolation_label": isolation_labels,
            "isolation_score": isolation_scores,
            "lof_label": lof_labels,
            "lof_score": lof_scores,
            "detection_category": categories,
        }
    )

    COMPARISON_ASSIGNMENTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    assignments.to_csv(
        COMPARISON_ASSIGNMENTS_PATH,
        index=False,
    )

    report = {
        "experiment": "isolation_forest_lof_comparison",
        "sample_count": len(positions),
        "feature_count": data.shape[1],
        "random_state": OUTLIER_RANDOM_STATE,
        "holdout_used": False,
        "isolation_forest": isolation_summary,
        "local_outlier_factor": lof_summary,
        "agreement": agreement,
    }

    COMPARISON_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Isolation Forest vs. LOF")
    print("------------------------")
    print(f"Samples: {agreement['samples']:,}")
    print(f"Isolation Forest outliers: {agreement['isolation_outliers']}")
    print(f"LOF outliers: {agreement['lof_outliers']}")

    print("\nAssignment agreement:")
    print(f"  Flagged by both: {agreement['both_outliers']}")
    print(f"  Isolation Forest only: {agreement['isolation_only']}")
    print(f"  LOF only: {agreement['lof_only']}")
    print(f"  Flagged by neither: {agreement['neither']}")

    if agreement["jaccard_similarity"] is not None:
        print("\nJaccard similarity: " f"{agreement['jaccard_similarity']:.4f}")

    print(
        "Expected random overlap: "
        f"{agreement['expected_overlap_under_random_selection']:.1f}"
    )

    print(f"\nSummary saved to: {COMPARISON_PATH}")
    print(f"Assignments saved to: {COMPARISON_ASSIGNMENTS_PATH}")


if __name__ == "__main__":
    main()
