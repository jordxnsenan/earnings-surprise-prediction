import json
import numpy as np
import pandas as pd

from pathlib import Path
from earnings_surprise.clustering.agglomerative import (
    AGGLOMERATIVE_LINKAGE,
    AGGLOMERATIVE_METRIC,
    run_agglomerative,
)
from earnings_surprise.clustering.prepare import (
    CLUSTER_RANDOM_STATE,
    CLUSTER_SAMPLE_SIZE,
    prepare_clustering_data,
)
from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.splits import HOLDOUT_START
from earnings_surprise.features import FEATURE_MATRIX_PATH

CLUSTER_COUNTS = (2, 3, 4, 5, 6, 7, 8)

AGGLOMERATIVE_SWEEP_PATH = Path("results/clustering/agglomerative_sweep.json")


def evaluate_agglomerative_counts(
    data: np.ndarray,
    cluster_counts: tuple[int, ...] = CLUSTER_COUNTS,
) -> list[dict]:
    """evaluates agglomerative clustering with several cluster counts"""
    if not cluster_counts:
        raise ValueError("At least one cluster count is required")

    if len(set(cluster_counts)) != len(cluster_counts):
        raise ValueError("Cluster counts must be unique")

    results = []

    for cluster_count in cluster_counts:
        _, summary = run_agglomerative(
            data,
            cluster_count=cluster_count,
        )

        results.append(summary)

    return results


def main() -> None:
    """compares agglomerative cluster counts on development data"""
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    transformed, selected_positions, _ = prepare_clustering_data(
        features,
        samples,
    )

    dates = pd.to_datetime(samples["target_date_pub"])

    selected_dates = dates.iloc[selected_positions]

    if (selected_dates >= HOLDOUT_START).any():
        raise ValueError("Holdout observations entered clustering")

    evaluations = evaluate_agglomerative_counts(transformed)

    report = {
        "experiment": "agglomerative_cluster_count_comparison",
        "sample_count": len(selected_positions),
        "feature_count": transformed.shape[1],
        "linkage": AGGLOMERATIVE_LINKAGE,
        "distance_metric": AGGLOMERATIVE_METRIC,
        "sampling": {
            "method": "random_without_replacement",
            "sample_size": CLUSTER_SAMPLE_SIZE,
            "random_state": CLUSTER_RANDOM_STATE,
            "development_samples": int((dates < HOLDOUT_START).sum()),
            "holdout_samples": int((dates >= HOLDOUT_START).sum()),
            "holdout_used": False,
        },
        "evaluations": evaluations,
    }

    AGGLOMERATIVE_SWEEP_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    AGGLOMERATIVE_SWEEP_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Agglomerative Cluster Count Comparison")
    print("---------------------------------------")

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

    print(f"\nResults saved to: {AGGLOMERATIVE_SWEEP_PATH}")


if __name__ == "__main__":
    main()