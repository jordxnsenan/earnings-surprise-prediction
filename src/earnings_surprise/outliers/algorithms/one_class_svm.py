import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.svm import OneClassSVM

from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.splits import HOLDOUT_START
from earnings_surprise.features import FEATURE_MATRIX_PATH
from earnings_surprise.outliers.prepare import (
    OUTLIER_COMPARISON_SAMPLE_SIZE,
    OUTLIER_RANDOM_STATE,
    prepare_outlier_data,
)

ONE_CLASS_NU = 0.05
ONE_CLASS_KERNEL = "rbf"
ONE_CLASS_GAMMA = "scale"

OUTLIER_LABEL = -1
SCORE_THRESHOLD = 0.0
TOP_ANOMALY_COUNT = 20
SCORE_PERCENTILES = (1, 5, 25, 50, 75, 95, 99)

ONE_CLASS_RESULTS_PATH = Path("results/outliers/one_class_svm_initial.json")

ONE_CLASS_ASSIGNMENTS_PATH = Path("results/outliers/one_class_svm_assignments.csv")


def run_one_class_svm(
    data: np.ndarray,
    nu: float = ONE_CLASS_NU,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """detects observations outside an RBF One-Class SVM boundary"""
    matrix = np.asarray(data, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 1:
        raise ValueError("Outlier data must be a valid 2D matrix")

    if not np.isfinite(matrix).all():
        raise ValueError("Outlier data must contain finite values")

    if (
        isinstance(nu, (bool, np.bool_))
        or not isinstance(nu, (int, float, np.integer, np.floating))
        or not np.isfinite(nu)
        or not 0 < nu <= 1
    ):
        raise ValueError("Invalid nu value")

    model = OneClassSVM(
        kernel=ONE_CLASS_KERNEL,
        gamma=ONE_CLASS_GAMMA,
        nu=float(nu),
    )

    model.fit(matrix)

    labels = model.predict(matrix)
    scores = model.decision_function(matrix).ravel()

    if not np.isfinite(scores).all():
        raise ValueError("One-Class SVM produced non-finite scores")

    outlier_count = int(np.sum(labels == OUTLIER_LABEL))

    summary = {
        "algorithm": "one_class_svm",
        "samples": len(matrix),
        "feature_count": matrix.shape[1],
        "kernel": ONE_CLASS_KERNEL,
        "gamma": ONE_CLASS_GAMMA,
        "nu": float(nu),
        "outlier_count": outlier_count,
        "inlier_count": int(len(matrix) - outlier_count),
        "outlier_fraction": outlier_count / len(matrix),
        "support_vector_count": len(model.support_),
        "score_type": "decision_function",
        "score_direction": "lower_is_more_anomalous",
        "score_threshold": SCORE_THRESHOLD,
        "score_percentiles": {
            str(percentile): float(np.percentile(scores, percentile))
            for percentile in SCORE_PERCENTILES
        },
    }

    return labels, scores, summary


def main() -> None:
    """evaluates One-Class SVM on development-only observations"""
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

    labels, scores, summary = run_one_class_svm(data)

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

    ONE_CLASS_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ONE_CLASS_RESULTS_PATH.write_text(
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
        ONE_CLASS_ASSIGNMENTS_PATH,
        index=False,
    )

    print("One-Class SVM — Initial Analysis")
    print("--------------------------------")

    print(f"Observations: {summary['samples']:,}")
    print(f"Features: {summary['feature_count']}")
    print(f"Kernel: {summary['kernel']}")
    print(f"Gamma: {summary['gamma']}")
    print(f"Nu: {summary['nu']:.2f}")

    print(f"\nOutliers: {summary['outlier_count']:,}")
    print(f"Inliers: {summary['inlier_count']:,}")
    print(f"Outlier fraction: {summary['outlier_fraction']:.2%}")
    print(f"Support vectors: {summary['support_vector_count']:,}")

    print("\nDecision-score percentiles:")

    for percentile, score in summary["score_percentiles"].items():
        print(f"  {percentile}th: {score:.4f}")

    print(f"\nResults saved to: {ONE_CLASS_RESULTS_PATH}")
    print(f"Assignments saved to: {ONE_CLASS_ASSIGNMENTS_PATH}")


if __name__ == "__main__":
    main()
