import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.decomposition import PCA

from earnings_surprise.clustering.kmeans import run_kmeans
from earnings_surprise.clustering.prepare import (
    CLUSTER_RANDOM_STATE,
    CLUSTER_SAMPLE_SIZE,
    prepare_clustering_data,
)
from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.splits import HOLDOUT_START
from earnings_surprise.features import FEATURE_MATRIX_PATH

KMEANS_CLUSTER_COUNT = 2

CLUSTER_RESULTS_DIRECTORY = Path("results/clustering")
CLUSTER_FIGURES_DIRECTORY = Path("results/figures")

PROFILE_COLUMNS = [
    "beat_rate4",
    "beat_streak4",
    "surprise_mean4",
    "surprise_std4",
    "actual_eps_lag1",
]


def plot_kmeans_pca(
    data: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> list[float]:
    """visualizes existing cluster assignments using two PCA components"""
    matrix = np.asarray(data, dtype=float)
    labels = np.asarray(labels)

    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("PCA requires at least two feature columns")

    if len(matrix) != len(labels):
        raise ValueError("Cluster labels and observations do not match")

    if not np.isfinite(matrix).all():
        raise ValueError("PCA input contains non-finite values")

    pca = PCA(n_components=2, svd_solver="full")
    projected = pca.fit_transform(matrix)

    fig, ax = plt.subplots(figsize=(10, 7))

    for cluster in np.unique(labels):
        cluster_mask = labels == cluster
        count = int(cluster_mask.sum())

        ax.scatter(
            projected[cluster_mask, 0],
            projected[cluster_mask, 1],
            s=15,
            alpha=0.5,
            label=f"Cluster {cluster} (n={count:,})",
        )

    explained_variance = pca.explained_variance_ratio_

    ax.set_xlabel(f"PC1 ({explained_variance[0]:.1%} variance explained)")

    ax.set_ylabel(f"PC2 ({explained_variance[1]:.1%} variance explained)")

    ax.set_title("K-Means Clusters — PCA Projection")
    ax.legend()
    ax.grid(alpha=0.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return [float(value) for value in explained_variance]


def summarize_cluster_profiles(
    features: pd.DataFrame,
    selected_positions: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """summarizes clusters using original historical feature values"""
    if len(selected_positions) != len(labels):
        raise ValueError("Cluster labels and selected rows do not match")

    missing_columns = [
        column for column in PROFILE_COLUMNS if column not in features.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing clustering profile columns: {missing_columns}")

    selected_features = features.iloc[selected_positions][PROFILE_COLUMNS].copy()

    selected_features["cluster"] = labels

    profile = (
        selected_features.groupby("cluster")
        .agg(
            samples=("beat_rate4", "size"),
            median_beat_rate4=("beat_rate4", "median"),
            median_beat_streak4=("beat_streak4", "median"),
            median_surprise_mean4=("surprise_mean4", "median"),
            median_surprise_std4=("surprise_std4", "median"),
            median_actual_eps_lag1=("actual_eps_lag1", "median"),
        )
        .reset_index()
    )

    return profile


def main() -> None:
    """visualizes and describes the two-cluster K-Means solution"""
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    transformed, selected_positions, _ = prepare_clustering_data(
        features,
        samples,
    )

    selected_dates = pd.to_datetime(samples.iloc[selected_positions]["target_date_pub"])

    if (selected_dates >= HOLDOUT_START).any():
        raise ValueError("Holdout observations entered clustering")

    labels, kmeans_summary = run_kmeans(
        transformed,
        cluster_count=KMEANS_CLUSTER_COUNT,
    )

    figure_path = CLUSTER_FIGURES_DIRECTORY / "kmeans_pca_k2.png"

    explained_variance = plot_kmeans_pca(
        transformed,
        labels,
        figure_path,
    )

    profile = summarize_cluster_profiles(
        features,
        selected_positions,
        labels,
    )

    CLUSTER_RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile_path = CLUSTER_RESULTS_DIRECTORY / "kmeans_profile_k2.csv"

    profile.to_csv(profile_path, index=False)

    assignments_path = CLUSTER_RESULTS_DIRECTORY / "kmeans_assignments_k2.csv"

    assignments = pd.DataFrame(
        {
            "source_row_position": selected_positions,
            "cluster": labels,
        }
    )

    assignments.to_csv(assignments_path, index=False)

    report = {
        **kmeans_summary,
        "sampling": {
            "method": "random_without_replacement",
            "random_state": CLUSTER_RANDOM_STATE,
            "sample_size": CLUSTER_SAMPLE_SIZE,
            "holdout_used": False,
        },
        "pca": {
            "explained_variance_ratio": explained_variance,
            "total_explained_variance": sum(explained_variance),
            "purpose": "visualization_only",
        },
    }

    report_path = CLUSTER_RESULTS_DIRECTORY / "kmeans_pca_k2.json"

    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("K-Means Cluster Interpretation")
    print("------------------------------")

    print(f"Samples: {len(labels):,}")
    print(f"Clusters: {KMEANS_CLUSTER_COUNT}")
    print(f"PCA explained variance: " f"{sum(explained_variance):.2%}")

    print("\nCluster Profiles:")
    print(profile.round(4).to_string(index=False))

    print(f"\nFigure saved to: {figure_path}")
    print(f"Profile saved to: {profile_path}")
    print(f"Assignments saved to: {assignments_path}")
    print(f"Summary saved to: {report_path}")


if __name__ == "__main__":
    main()
