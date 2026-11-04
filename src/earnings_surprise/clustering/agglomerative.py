import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from earnings_surprise.clustering.prepare import (
    CLUSTER_RANDOM_STATE,
    CLUSTER_SAMPLE_SIZE,
    prepare_clustering_data,
)
from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.splits import HOLDOUT_START
from earnings_surprise.features import FEATURE_MATRIX_PATH

INITIAL_CLUSTER_COUNT = 2
AGGLOMERATIVE_LINKAGE = "ward"
AGGLOMERATIVE_METRIC = "euclidean"
SILHOUETTE_SAMPLE_SIZE = 1000

AGGLOMERATIVE_RESULTS_PATH = Path("results/clustering/agglomerative_initial.json")

AGGLOMERATIVE_ASSIGNMENTS_PATH = Path(
    "results/clustering/agglomerative_assignments_k2.csv"
)


def run_agglomerative(
    data: np.ndarray,
    cluster_count: int = INITIAL_CLUSTER_COUNT,
) -> tuple[np.ndarray, dict]:
    """fits agglomerative clustering and summarizes its performance"""
    matrix = np.asarray(data, dtype=float)

    if matrix.ndim != 2 or len(matrix) < 3:
        raise ValueError("Clustering data must be a 2D matrix with at least 3 rows")

    if not np.isfinite(matrix).all():
        raise ValueError("Clustering data must contain finite values")

    if (
        isinstance(cluster_count, (bool, np.bool_))
        or not isinstance(cluster_count, (int, np.integer))
        or not 2 <= cluster_count < len(matrix)
    ):
        raise ValueError("Invalid number of clusters")

    model = AgglomerativeClustering(
        n_clusters=int(cluster_count),
        linkage=AGGLOMERATIVE_LINKAGE,
        metric=AGGLOMERATIVE_METRIC,
    )

    labels = model.fit_predict(matrix)

    unique_labels, counts = np.unique(
        labels,
        return_counts=True,
    )

    silhouette_count = min(
        SILHOUETTE_SAMPLE_SIZE,
        len(matrix),
    )

    summary = {
        "algorithm": "agglomerative",
        "linkage": AGGLOMERATIVE_LINKAGE,
        "distance_metric": AGGLOMERATIVE_METRIC,
        "samples": len(matrix),
        "feature_count": matrix.shape[1],
        "cluster_count": len(unique_labels),
        "silhouette_score": float(
            silhouette_score(
                matrix,
                labels,
                sample_size=silhouette_count,
                random_state=CLUSTER_RANDOM_STATE,
            )
        ),
        "silhouette_sample_size": silhouette_count,
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
    """runs agglomerative clustering on the development sample"""
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    transformed, selected_positions, _ = prepare_clustering_data(
        features,
        samples,
    )

    selected_dates = pd.to_datetime(samples.iloc[selected_positions]["target_date_pub"])

    if (selected_dates >= HOLDOUT_START).any():
        raise ValueError("Holdout observations entered clustering")

    labels, summary = run_agglomerative(transformed)

    all_dates = pd.to_datetime(samples["target_date_pub"])

    summary["sampling"] = {
        "method": "random_without_replacement",
        "random_state": CLUSTER_RANDOM_STATE,
        "sample_size": CLUSTER_SAMPLE_SIZE,
        "development_samples": int((all_dates < HOLDOUT_START).sum()),
        "holdout_samples": int((all_dates >= HOLDOUT_START).sum()),
        "holdout_used": False,
    }

    AGGLOMERATIVE_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    AGGLOMERATIVE_RESULTS_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    assignments = pd.DataFrame(
        {
            "source_row_position": selected_positions,
            "cluster": labels,
        }
    )

    assignments.to_csv(
        AGGLOMERATIVE_ASSIGNMENTS_PATH,
        index=False,
    )

    print("Initial Agglomerative Clustering")
    print("--------------------------------")
    print(f"Samples: {summary['samples']:,}")
    print(f"Features: {summary['feature_count']}")
    print(f"Clusters: {summary['cluster_count']}")
    print(f"Linkage: {summary['linkage']}")
    print(f"Silhouette Score: {summary['silhouette_score']:.4f}")
    print("Calinski-Harabasz Index: " f"{summary['calinski_harabasz_score']:.2f}")
    print("Davies-Bouldin Index: " f"{summary['davies_bouldin_score']:.4f}")

    print("\nCluster sizes:")

    for label, count in summary["cluster_sizes"].items():
        print(f"  Cluster {label}: {count:,}")

    print(f"\nResults saved to: {AGGLOMERATIVE_RESULTS_PATH}")
    print(f"Assignments saved to: {AGGLOMERATIVE_ASSIGNMENTS_PATH}")


if __name__ == "__main__":
    main()
