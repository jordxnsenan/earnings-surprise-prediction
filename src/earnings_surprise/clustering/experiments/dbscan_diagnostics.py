import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.neighbors import NearestNeighbors

from earnings_surprise.clustering.prepare import (
    prepare_clustering_data,
)
from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.features import FEATURE_MATRIX_PATH

MIN_SAMPLES_OPTIONS = (10, 40)
DISTANCE_PERCENTILES = (50, 70, 85, 95)

DBSCAN_DIAGNOSTICS_PATH = Path("results/clustering/dbscan_distances.json")


def summarize_neighbor_distances(
    data: np.ndarray,
    min_samples: int,
) -> dict:
    """summarizes nearest-neighbor distances"""
    matrix = np.asarray(data, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("Clustering data must be two-dimensional")

    if not np.isfinite(matrix).all():
        raise ValueError("Clustering data must contain finite values")

    if (
        isinstance(min_samples, bool)
        or not isinstance(min_samples, int)
        or min_samples < 2
        or min_samples > len(matrix)
    ):
        raise ValueError("Invalid min_samples value")

    neighbors = NearestNeighbors(
        n_neighbors=min_samples,
        metric="euclidean",
    )

    neighbors.fit(matrix)

    distances, _ = neighbors.kneighbors(matrix)

    # ensures each observation is included in its own neighborhood
    kth_distances = distances[:, -1]

    percentiles = {
        str(percentile): float(np.percentile(kth_distances, percentile))
        for percentile in DISTANCE_PERCENTILES
    }

    return {
        "min_samples": min_samples,
        "samples": len(matrix),
        "feature_count": matrix.shape[1],
        "distance_percentiles": percentiles,
        "minimum_distance": float(kth_distances.min()),
        "maximum_distance": float(kth_distances.max()),
    }


def main() -> None:
    """inspects DBSCAN neighborhoods in the development sample"""
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    transformed, selected_positions, _ = prepare_clustering_data(
        features,
        samples,
    )

    diagnostics = [
        summarize_neighbor_distances(transformed, min_samples)
        for min_samples in MIN_SAMPLES_OPTIONS
    ]

    report = {
        "experiment": "dbscan_neighbor_distances",
        "sample_count": len(selected_positions),
        "feature_count": transformed.shape[1],
        "holdout_used": False,
        "diagnostics": diagnostics,
    }

    DBSCAN_DIAGNOSTICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    DBSCAN_DIAGNOSTICS_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    for result in diagnostics:
        print(f"\nmin_samples = {result['min_samples']}")

        for percentile, distance in result["distance_percentiles"].items():
            print(f"  {percentile}th percentile: " f"{distance:.4f}")

    print(f"\nSaved to: {DBSCAN_DIAGNOSTICS_PATH}")


if __name__ == "__main__":
    main()