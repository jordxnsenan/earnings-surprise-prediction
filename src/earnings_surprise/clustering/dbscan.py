import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

NOISE_LABEL = -1
MINIMUM_NEIGHBORS = 2
MINIMUM_CLUSTER_SIZE_FOR_METRICS = 2
SILHOUETTE_SAMPLE_SIZE = 1000
CLUSTER_RANDOM_STATE = 42


def select_silhouette_sample(labels: np.ndarray) -> np.ndarray | None:
    """samples clustered observations while representing every cluster"""
    sample_count = min(SILHOUETTE_SAMPLE_SIZE, len(labels))

    if sample_count == len(labels):
        return np.arange(len(labels))

    clusters = np.unique(labels)

    if len(clusters) * MINIMUM_CLUSTER_SIZE_FOR_METRICS > sample_count:
        return None

    generator = np.random.default_rng(CLUSTER_RANDOM_STATE)

    required_indices = []

    for cluster in clusters:
        cluster_indices = np.flatnonzero(labels == cluster)

        selected = generator.choice(
            cluster_indices,
            size=MINIMUM_CLUSTER_SIZE_FOR_METRICS,
            replace=False,
        )

        required_indices.extend(selected)

    available_mask = np.ones(len(labels), dtype=bool)
    available_mask[required_indices] = False

    additional_count = sample_count - len(required_indices)

    additional_indices = generator.choice(
        np.flatnonzero(available_mask),
        size=additional_count,
        replace=False,
    )

    return np.sort(np.concatenate([required_indices, additional_indices]))


def run_dbscan(
    data: np.ndarray,
    eps: float,
    min_samples: int,
) -> tuple[np.ndarray, dict]:
    """fits DBSCAN and evaluates the resulting clusters"""
    matrix = np.asarray(data, dtype=float)

    if matrix.ndim != 2 or len(matrix) < MINIMUM_NEIGHBORS:
        raise ValueError("Clustering data must be a nonempty 2D matrix")

    if not np.isfinite(matrix).all():
        raise ValueError("Clustering data must contain finite values")

    if isinstance(eps, (bool, np.bool_)) or not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite number")

    if (
        isinstance(min_samples, (bool, np.bool_))
        or not isinstance(min_samples, (int, np.integer))
        or not MINIMUM_NEIGHBORS <= min_samples <= len(matrix)
    ):
        raise ValueError("Invalid min_samples value")

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean",
        n_jobs=1,
    )

    labels = model.fit_predict(matrix)

    noise_mask = labels == NOISE_LABEL
    clustered_mask = ~noise_mask

    clustered_labels = labels[clustered_mask]
    clustered_data = matrix[clustered_mask]

    unique_labels, counts = np.unique(
        clustered_labels,
        return_counts=True,
    )

    cluster_count = len(unique_labels)
    noise_count = int(noise_mask.sum())
    clustered_count = len(clustered_data)

    summary = {
        "algorithm": "dbscan",
        "samples": len(matrix),
        "feature_count": matrix.shape[1],
        "eps": float(eps),
        "min_samples": int(min_samples),
        "cluster_count": cluster_count,
        "clustered_samples": clustered_count,
        "clustered_fraction": clustered_count / len(matrix),
        "noise_samples": noise_count,
        "noise_fraction": noise_count / len(matrix),
        "core_samples": len(model.core_sample_indices_),
        "cluster_sizes": {
            str(label): int(count)
            for label, count in zip(
                unique_labels,
                counts,
                strict=True,
            )
        },
        "evaluation_population": "clustered_samples_only",
        "silhouette_score": None,
        "silhouette_sample_size": None,
        "calinski_harabasz_score": None,
        "davies_bouldin_score": None,
    }

    # internal metrics require at least two non-degenerate clusters
    has_valid_clusters = (
        cluster_count >= 2
        and cluster_count < clustered_count
        and counts.min() >= MINIMUM_CLUSTER_SIZE_FOR_METRICS
    )

    if has_valid_clusters:
        selected_indices = select_silhouette_sample(clustered_labels)

        if selected_indices is not None:
            summary["silhouette_score"] = float(
                silhouette_score(
                    clustered_data[selected_indices],
                    clustered_labels[selected_indices],
                )
            )

            summary["silhouette_sample_size"] = len(selected_indices)

        summary["calinski_harabasz_score"] = float(
            calinski_harabasz_score(
                clustered_data,
                clustered_labels,
            )
        )

        summary["davies_bouldin_score"] = float(
            davies_bouldin_score(
                clustered_data,
                clustered_labels,
            )
        )

    return labels, summary