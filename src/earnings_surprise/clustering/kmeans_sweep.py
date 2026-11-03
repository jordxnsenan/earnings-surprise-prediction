import json
import numpy as np
import pandas as pd

from pathlib import Path

from earnings_surprise.clustering.kmeans import run_kmeans
from earnings_surprise.clustering.prepare import (
    CLUSTER_RANDOM_STATE,
    CLUSTER_SAMPLE_SIZE,
    prepare_clustering_data,
)
from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.splits import HOLDOUT_START
from earnings_surprise.features import FEATURE_MATRIX_PATH

CLUSTER_COUNTS = (2, 3, 4, 5, 6, 7, 8)

KMEANS_SWEEP_PATH = Path("results/clustering/kmeans_sweep.json")


def evaluate_cluster_counts(
    data: np.ndarray,
    cluster_counts: tuple[int, ...] = CLUSTER_COUNTS,
) -> list[dict]:
    """evaluates K-Means using several cluster counts on the same data"""
    if not cluster_counts:
        raise ValueError("At least one cluster count is required")

    if len(set(cluster_counts)) != len(cluster_counts):
        raise ValueError("Cluster counts must be unique")

    results = []

    for cluster_count in cluster_counts:
        _, summary = run_kmeans(
            data,
            cluster_count=cluster_count,
        )

        results.append(summary)

    return results


def main() -> None:
    """compares K-Means cluster counts using development observations"""
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    transformed, selected_positions, _ = prepare_clustering_data(
        features,
        samples,
    )

    evaluations = evaluate_cluster_counts(transformed)

    dates = pd.to_datetime(samples["target_date_pub"])

    report = {
        "experiment": "kmeans_cluster_count_comparison",
        "sample_count": len(selected_positions),
        "feature_count": transformed.shape[1],
        "sampling": {
            "method": "random_without_replacement",
            "sample_size": CLUSTER_SAMPLE_SIZE,
            "random_state": CLUSTER_RANDOM_STATE,
            "holdout_samples": int((dates >= HOLDOUT_START).sum()),
            "holdout_used": False,
        },
        "evaluations": evaluations,
    }

    KMEANS_SWEEP_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    KMEANS_SWEEP_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("K-Means Cluster Count Comparison")
    print("--------------------------------")

    print(
        f"{'Clusters':>8} "
        f"{'Silhouette':>12} "
        f"{'Calinski-Harabasz':>20} "
        f"{'Davies-Bouldin':>18}"
    )

    for result in evaluations:
        print(
            f"{result['cluster_count']:>8} "
            f"{result['silhouette_score']:>12.4f} "
            f"{result['calinski_harabasz_score']:>20.2f} "
            f"{result['davies_bouldin_score']:>18.4f}"
        )

    print(f"\nResults saved to: {KMEANS_SWEEP_PATH}")


if __name__ == "__main__":
    main()