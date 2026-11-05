import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.neighbors import LocalOutlierFactor

from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.splits import HOLDOUT_START
from earnings_surprise.features import FEATURE_MATRIX_PATH
from earnings_surprise.outliers.prepare import (
    OUTLIER_COMPARISON_SAMPLE_SIZE,
    OUTLIER_RANDOM_STATE,
    prepare_outlier_data,
)

LOF_NEIGHBOR_COUNT = 20
LOF_CONTAMINATION = 0.05
LOF_DISTANCE_METRIC = "euclidean"
OUTLIER_LABEL = -1

SCORE_PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
TOP_ANOMALY_COUNT = 20

LOF_RESULTS_PATH = Path("results/outliers/lof_initial.json")

LOF_ASSIGNMENTS_PATH = Path("results/outliers/lof_assignments.csv")


def run_local_outlier_factor(
    data: np.ndarray,
    neighbor_count: int = LOF_NEIGHBOR_COUNT,
    contamination: float = LOF_CONTAMINATION,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """detects observations with unusually low local density"""
    matrix = np.asarray(data, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 1:
        raise ValueError("Outlier data must be a valid 2D matrix")

    if not np.isfinite(matrix).all():
        raise ValueError("Outlier data must contain finite values")

    if (
        isinstance(neighbor_count, (bool, np.bool_))
        or not isinstance(neighbor_count, (int, np.integer))
        or not 2 <= neighbor_count < len(matrix)
    ):
        raise ValueError("Invalid neighbor count")

    if (
        isinstance(contamination, (bool, np.bool_))
        or not isinstance(
            contamination,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(contamination)
        or not 0 < contamination <= 0.5
    ):
        raise ValueError("Invalid contamination value")

    model = LocalOutlierFactor(
        n_neighbors=int(neighbor_count),
        contamination=float(contamination),
        metric=LOF_DISTANCE_METRIC,
        n_jobs=-1,
    )

    labels = model.fit_predict(matrix)
    scores = model.negative_outlier_factor_.copy()

    if not np.isfinite(scores).all():
        raise ValueError("LOF produced non-finite anomaly scores")

    outlier_count = int(np.sum(labels == OUTLIER_LABEL))

    summary = {
        "algorithm": "local_outlier_factor",
        "samples": len(matrix),
        "feature_count": matrix.shape[1],
        "neighbor_count": int(model.n_neighbors_),
        "distance_metric": LOF_DISTANCE_METRIC,
        "contamination": float(contamination),
        "outlier_count": outlier_count,
        "inlier_count": int(len(matrix) - outlier_count),
        "outlier_fraction": outlier_count / len(matrix),
        "score_direction": "lower_is_more_anomalous",
        "score_cutoff": float(model.offset_),
        "score_percentiles": {
            str(percentile): float(np.percentile(scores, percentile))
            for percentile in SCORE_PERCENTILES
        },
    }

    return labels, scores, summary


def main() -> None:
    """evaluates LOF on a development-only sample"""
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    data, positions, _ = prepare_outlier_data(
        features,
        samples,
        sample_size=OUTLIER_COMPARISON_SAMPLE_SIZE,
    )

    dates = pd.to_datetime(samples["target_date_pub"])
    selected_dates = dates.iloc[positions]

    if (selected_dates >= HOLDOUT_START).any():
        raise ValueError("Holdout observations entered outlier detection")

    labels, scores, summary = run_local_outlier_factor(data)

    summary["sampling"] = {
        "method": "random_without_replacement",
        "sample_size": len(positions),
        "random_state": OUTLIER_RANDOM_STATE,
        "development_samples": int((dates < HOLDOUT_START).sum()),
        "holdout_samples": int((dates >= HOLDOUT_START).sum()),
        "holdout_used": False,
    }

    most_anomalous = np.argsort(scores)[:TOP_ANOMALY_COUNT]

    summary["most_anomalous_observations"] = [
        {
            "source_row_position": int(positions[index]),
            "anomaly_score": float(scores[index]),
        }
        for index in most_anomalous
    ]

    LOF_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOF_RESULTS_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    assignments = pd.DataFrame(
        {
            "source_row_position": positions,
            "anomaly_label": labels,
            "anomaly_score": scores,
        }
    )

    assignments.to_csv(
        LOF_ASSIGNMENTS_PATH,
        index=False,
    )

    print("Local Outlier Factor — Initial Analysis")
    print("---------------------------------------")
    print(f"Observations: {summary['samples']:,}")
    print(f"Features: {summary['feature_count']}")
    print(f"Neighbors: {summary['neighbor_count']}")
    print(f"Contamination: {summary['contamination']:.1%}")

    print(f"\nOutliers: {summary['outlier_count']:,}")
    print(f"Inliers: {summary['inlier_count']:,}")
    print(f"Outlier fraction: {summary['outlier_fraction']:.2%}")
    print(f"Score cutoff: {summary['score_cutoff']:.4f}")

    print("\nAnomaly score percentiles:")

    for percentile, score in summary["score_percentiles"].items():
        print(f"  {percentile}th: {score:.4f}")

    print(f"\nResults saved to: {LOF_RESULTS_PATH}")
    print(f"Assignments saved to: {LOF_ASSIGNMENTS_PATH}")


if __name__ == "__main__":
    main()
