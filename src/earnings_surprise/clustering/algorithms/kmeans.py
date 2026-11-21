import json
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.splits import HOLDOUT_START
from earnings_surprise.features import FEATURE_MATRIX_PATH
from earnings_surprise.clustering.prepare import (
    CLUSTER_SAMPLE_SIZE,
    prepare_clustering_data,
)

INITIAL_CLUSTER_COUNT = 4
CLUSTER_RANDOM_STATE = 42
KMEANS_INITIALIZATIONS = 10
SILHOUETTE_SAMPLE_SIZE = 1000

KMEANS_RESULTS_PATH = Path("results/clustering/kmeans_initial.json")


def run_kmeans(
    data: np.ndarray,
    cluster_count: int = INITIAL_CLUSTER_COUNT,
) -> tuple[np.ndarray, dict]:
    """fits K-Means and summarizes the results."""
    matrix = np.asarray(data, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("Clustering data must be two-dimensional")

    if not np.isfinite(matrix).all():
        raise ValueError("Clustering data must contain finite values")

    if not 2 <= cluster_count < len(matrix):
        raise ValueError("Invalid number of clusters")

    model = KMeans(
        n_clusters=cluster_count,
        n_init=KMEANS_INITIALIZATIONS,
        random_state=CLUSTER_RANDOM_STATE,
    )

    labels = model.fit_predict(matrix)

    unique_labels, counts = np.unique(
        labels,
        return_counts=True,
    )

    if len(unique_labels) != cluster_count:
        raise ValueError("K-Means produced fewer clusters than requested")

    silhouette_sample_count = min(
        SILHOUETTE_SAMPLE_SIZE,
        len(matrix),
    )

    summary = {
        "algorithm": "kmeans",
        "samples": len(matrix),
        "feature_count": matrix.shape[1],
        "cluster_count": cluster_count,
        "inertia": float(model.inertia_),
        "silhouette_score": float(
            silhouette_score(
                matrix,
                labels,
                sample_size=silhouette_sample_count,
                random_state=CLUSTER_RANDOM_STATE,
            )
        ),
        "silhouette_sample_size": silhouette_sample_count,
        "calinski_harabasz_score": float(calinski_harabasz_score(matrix, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(matrix, labels)),
        "cluster_sizes": {
            str(label): int(count)
            for label, count in zip(
                unique_labels,
                counts,
                strict=True,
            )
        },
    }

    return labels, summary


def main() -> None:
    """runs initial K-Means experiment on development data"""
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    transformed, selected_positions, preprocessor = prepare_clustering_data(
        features,
        samples,
    )

    labels, summary = run_kmeans(transformed)

    dates = pd.to_datetime(samples["target_date_pub"])

    summary["sampling"] = {
        "method": "random_without_replacement",
        "random_state": CLUSTER_RANDOM_STATE,
        "requested_samples": CLUSTER_SAMPLE_SIZE,
        "development_samples": int((dates < HOLDOUT_START).sum()),
        "holdout_samples": int((dates >= HOLDOUT_START).sum()),
        "holdout_used": False,
    }

    summary["transformed_feature_names"] = list(preprocessor.get_feature_names_out())

    sampled_years = dates.iloc[selected_positions].dt.year

    summary["sampled_year_counts"] = {
        str(year): int(count)
        for year, count in sampled_years.value_counts().sort_index().items()
    }

    KMEANS_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    KMEANS_RESULTS_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Initial K-Means Clustering")
    print("--------------------------")
    print(f"Samples: {summary['samples']:,}")
    print(f"Features: {summary['feature_count']}")
    print(f"Clusters: {summary['cluster_count']}")
    print(f"Silhouette Score: {summary['silhouette_score']:.4f}")
    print("Calinski-Harabasz Index: " f"{summary['calinski_harabasz_score']:.2f}")
    print("Davies-Bouldin Index: " f"{summary['davies_bouldin_score']:.4f}")

    print("\nCluster sizes:")
    for label, count in summary["cluster_sizes"].items():
        print(f"  Cluster {label}: {count:,}")

    print(f"\nResults saved to: {KMEANS_RESULTS_PATH}")


if __name__ == "__main__":
    main()