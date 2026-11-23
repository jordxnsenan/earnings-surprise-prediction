import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import IsolationForest

from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.splits import HOLDOUT_START
from earnings_surprise.features import FEATURE_MATRIX_PATH
from earnings_surprise.outliers.prepare import (
    OUTLIER_RANDOM_STATE,
    prepare_outlier_data,
)

ISOLATION_TREE_COUNT = 200
ISOLATION_MAX_SAMPLES = 256
ISOLATION_CONTAMINATION = 0.05

ISOLATION_OUTLIER_LABEL = -1
TOP_ANOMALY_COUNT = 20

SCORE_PERCENTILES = (1, 5, 25, 50, 75, 95, 99)

ISOLATION_RESULTS_PATH = Path("results/outliers/isolation_forest_initial.json")


def run_isolation_forest(
    data: np.ndarray,
    contamination: float = ISOLATION_CONTAMINATION,
    tree_count: int = ISOLATION_TREE_COUNT,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """fits Isolation Forest and summarizes detected anomalies"""
    matrix = np.asarray(data, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("Outlier data must be a valid 2D matrix")

    if not np.isfinite(matrix).all():
        raise ValueError("Outlier data must contain finite values")

    if (
        isinstance(contamination, (bool, np.bool_))
        or not isinstance(contamination, (int, float, np.floating))
        or not np.isfinite(contamination)
        or not 0 < contamination <= 0.5
    ):
        raise ValueError("Invalid contamination value")

    if (
        isinstance(tree_count, (bool, np.bool_))
        or not isinstance(tree_count, (int, np.integer))
        or tree_count < 1
    ):
        raise ValueError("Invalid tree count")

    model = IsolationForest(
        n_estimators=int(tree_count),
        max_samples=min(ISOLATION_MAX_SAMPLES, len(matrix)),
        contamination=float(contamination),
        random_state=OUTLIER_RANDOM_STATE,
        n_jobs=-1,
    )

    labels = model.fit_predict(matrix)
    scores = model.score_samples(matrix)

    if not np.isfinite(scores).all():
        raise ValueError("Isolation Forest produced non-finite scores")

    outlier_mask = labels == ISOLATION_OUTLIER_LABEL
    outlier_count = int(outlier_mask.sum())

    score_percentiles = {
        str(percentile): float(np.percentile(scores, percentile))
        for percentile in SCORE_PERCENTILES
    }

    summary = {
        "algorithm": "isolation_forest",
        "samples": len(matrix),
        "feature_count": matrix.shape[1],
        "tree_count": int(tree_count),
        "max_samples_per_tree": int(model.max_samples_),
        "contamination": float(contamination),
        "outlier_count": outlier_count,
        "inlier_count": int(len(matrix) - outlier_count),
        "outlier_fraction": outlier_count / len(matrix),
        "score_direction": "lower_is_more_anomalous",
        "score_percentiles": score_percentiles,
    }

    return labels, scores, summary


def main() -> None:
    """detects anomalies in the full development population"""
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    transformed, selected_positions, preprocessor = prepare_outlier_data(
        features, samples
    )

    labels, scores, summary = run_isolation_forest(transformed)

    dates = pd.to_datetime(samples["target_date_pub"])
    selected_dates = dates.iloc[selected_positions]

    if (selected_dates >= HOLDOUT_START).any():
        raise ValueError("Holdout observations entered outlier detection")

    # preserves the source positions of the most unusual observations
    most_anomalous_indices = np.argsort(scores)[:TOP_ANOMALY_COUNT]

    summary["most_anomalous_observations"] = [
        {
            "source_row_position": int(selected_positions[index]),
            "anomaly_score": float(scores[index]),
        }
        for index in most_anomalous_indices
    ]

    summary["development"] = {
        "first_date": selected_dates.min().date().isoformat(),
        "last_date": selected_dates.max().date().isoformat(),
        "sample_count": len(selected_positions),
        "holdout_samples": int((dates >= HOLDOUT_START).sum()),
        "holdout_used": False,
    }

    summary["transformed_feature_names"] = [
        str(name) for name in preprocessor.get_feature_names_out()
    ]

    ISOLATION_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ISOLATION_RESULTS_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Isolation Forest — Initial Outlier Analysis")
    print("-------------------------------------------")

    print(f"Observations: {summary['samples']:,}")
    print(f"Features: {summary['feature_count']}")
    print(f"Trees: {summary['tree_count']}")
    print(f"Contamination: {summary['contamination']:.1%}")

    print(f"\nFlagged outliers: {summary['outlier_count']:,}")
    print(f"Retained inliers: {summary['inlier_count']:,}")
    print(f"Outlier fraction: {summary['outlier_fraction']:.2%}")

    print("\nAnomaly score percentiles:")

    for percentile, score in summary["score_percentiles"].items():
        print(f"  {percentile}th: {score:.4f}")

    print(
        "\nReserved holdout observations: "
        f"{summary['development']['holdout_samples']:,}"
    )

    print(f"\nResults saved to: {ISOLATION_RESULTS_PATH}")


if __name__ == "__main__":
    main()
