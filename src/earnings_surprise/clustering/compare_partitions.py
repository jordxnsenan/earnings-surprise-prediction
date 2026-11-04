import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from earnings_surprise.clustering.prepare import prepare_clustering_data
from earnings_surprise.clustering.visualize import summarize_cluster_profiles
from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.features import FEATURE_MATRIX_PATH

KMEANS_ASSIGNMENTS_PATH = Path("results/clustering/kmeans_assignments_k2.csv")

AGGLOMERATIVE_ASSIGNMENTS_PATH = Path(
    "results/clustering/agglomerative_assignments_k2.csv"
)

COMPARISON_RESULTS_PATH = Path("results/clustering/partition_comparison.json")

AGGLOMERATIVE_PROFILE_PATH = Path("results/clustering/agglomerative_profile_k2.csv")

COMPARISON_FIGURE_PATH = Path("results/figures/kmeans_vs_agglomerative_pca.png")

CLUSTER_COUNT = 2
FIGURE_DPI = 250


def align_cluster_labels(
    selected_positions: np.ndarray,
    assignments: pd.DataFrame,
) -> np.ndarray:
    """aligns saved cluster labels with the sampled observation order"""
    required_columns = {"source_row_position", "cluster"}

    if not required_columns.issubset(assignments.columns):
        raise ValueError("Missing cluster assignment columns")

    if assignments["source_row_position"].duplicated().any():
        raise ValueError("Duplicate source row positions")

    if len(assignments) != len(selected_positions) or set(
        assignments["source_row_position"]
    ) != set(selected_positions):
        raise ValueError("Cluster assignments do not match sampled rows")

    ordered = assignments.set_index("source_row_position").loc[selected_positions]

    labels = pd.to_numeric(
        ordered["cluster"],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.isfinite(labels).all() or not np.array_equal(labels, np.rint(labels)):
        raise ValueError("Cluster labels must be finite integers")

    labels = labels.astype(int)

    if len(np.unique(labels)) != CLUSTER_COUNT:
        raise ValueError("Expected exactly two clusters")

    return labels


def compare_partitions(
    kmeans_labels: np.ndarray,
    agglomerative_labels: np.ndarray,
) -> dict:
    """measures agreement between two clustering assignments"""
    if len(kmeans_labels) != len(agglomerative_labels):
        raise ValueError("Cluster label counts do not match")

    if len(kmeans_labels) == 0:
        raise ValueError("Cluster labels cannot be empty")

    contingency = pd.crosstab(
        pd.Series(kmeans_labels, name="kmeans"),
        pd.Series(agglomerative_labels, name="agglomerative"),
    )

    return {
        "adjusted_rand_index": float(
            adjusted_rand_score(
                kmeans_labels,
                agglomerative_labels,
            )
        ),
        "contingency_matrix": contingency.to_numpy().tolist(),
        "kmeans_cluster_ids": [int(value) for value in contingency.index],
        "agglomerative_cluster_ids": [int(value) for value in contingency.columns],
    }


def plot_partition_comparison(
    data: np.ndarray,
    kmeans_labels: np.ndarray,
    agglomerative_labels: np.ndarray,
    output_path: Path,
) -> list[float]:
    """shows two cluster assignments using a shared PCA projection"""
    matrix = np.asarray(data, dtype=float)

    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("PCA requires a two-dimensional feature matrix")

    if len(matrix) != len(kmeans_labels) or len(matrix) != len(agglomerative_labels):
        raise ValueError("Cluster labels and observations do not match")

    if not np.isfinite(matrix).all():
        raise ValueError("PCA data contains non-finite values")

    pca = PCA(n_components=2, svd_solver="full")
    projected = pca.fit_transform(matrix)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    configurations = [
        ("K-Means (k=2)", kmeans_labels),
        ("Agglomerative (Ward, k=2)", agglomerative_labels),
    ]

    for ax, (title, labels) in zip(
        axes,
        configurations,
        strict=True,
    ):
        for cluster in np.unique(labels):
            mask = labels == cluster

            ax.scatter(
                projected[mask, 0],
                projected[mask, 1],
                s=12,
                alpha=0.45,
                label=f"Cluster {cluster} (n={mask.sum():,})",
            )

        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.grid(alpha=0.2)
        ax.legend()

    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")

    fig.suptitle("Clustering Assignment Comparison")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    return [float(value) for value in pca.explained_variance_ratio_]


def main() -> None:
    """compares saved K-Means and Agglomerative assignments"""
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    transformed, selected_positions, _ = prepare_clustering_data(
        features,
        samples,
    )

    kmeans_assignments = pd.read_csv(KMEANS_ASSIGNMENTS_PATH)
    agglomerative_assignments = pd.read_csv(AGGLOMERATIVE_ASSIGNMENTS_PATH)

    kmeans_labels = align_cluster_labels(
        selected_positions,
        kmeans_assignments,
    )

    agglomerative_labels = align_cluster_labels(
        selected_positions,
        agglomerative_assignments,
    )

    comparison = compare_partitions(
        kmeans_labels,
        agglomerative_labels,
    )

    explained_variance = plot_partition_comparison(
        transformed,
        kmeans_labels,
        agglomerative_labels,
        COMPARISON_FIGURE_PATH,
    )

    profile = summarize_cluster_profiles(
        features,
        selected_positions,
        agglomerative_labels,
    )

    AGGLOMERATIVE_PROFILE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile.to_csv(AGGLOMERATIVE_PROFILE_PATH, index=False)

    report = {
        "experiment": "kmeans_agglomerative_comparison",
        "samples": len(selected_positions),
        "feature_count": transformed.shape[1],
        "holdout_used": False,
        **comparison,
        "pca": {
            "explained_variance_ratio": explained_variance,
            "total_explained_variance": sum(explained_variance),
            "purpose": "visualization_only",
        },
    }

    COMPARISON_RESULTS_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("K-Means and Agglomerative Comparison")
    print("------------------------------------")
    print(f"Samples: {report['samples']:,}")
    print(f"Adjusted Rand Index: {report['adjusted_rand_index']:.4f}")
    print(
        f"PCA explained variance: " f"{report['pca']['total_explained_variance']:.2%}"
    )

    print("\nCluster agreement table:")
    print(
        pd.DataFrame(
            comparison["contingency_matrix"],
            index=comparison["kmeans_cluster_ids"],
            columns=comparison["agglomerative_cluster_ids"],
        )
    )

    print("\nAgglomerative cluster profiles:")
    print(profile.round(4).to_string(index=False))

    print(f"\nFigure saved to: {COMPARISON_FIGURE_PATH}")
    print(f"Report saved to: {COMPARISON_RESULTS_PATH}")


if __name__ == "__main__":
    main()
