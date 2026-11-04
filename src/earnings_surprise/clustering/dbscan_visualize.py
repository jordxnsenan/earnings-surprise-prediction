import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.decomposition import PCA

from earnings_surprise.clustering.dbscan import (
    NOISE_LABEL,
    run_dbscan,
)
from earnings_surprise.clustering.dbscan_sweep import (
    DBSCAN_SWEEP_PATH,
)
from earnings_surprise.clustering.prepare import (
    prepare_clustering_data,
)
from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.features import FEATURE_MATRIX_PATH

DBSCAN_FIGURE_PATH = Path("results/figures/dbscan_pca_noise_comparison.png")

DBSCAN_VISUALIZATION_PATH = Path("results/clustering/dbscan_visualization.json")

COMPARISON_SETTINGS = ((10, 50), (10, 85))
FIGURE_DPI = 250


def select_dbscan_configurations(sweep: dict) -> list[dict]:
    """selects two DBSCAN configs. from the saved parameter sweep"""
    if sweep.get("holdout_used") is not False:
        raise ValueError("DBSCAN sweep must exclude the holdout")

    selected = []

    for min_samples, percentile in COMPARISON_SETTINGS:
        matches = [
            result
            for result in sweep["evaluations"]
            if result["min_samples"] == min_samples
            and result["eps_percentile"] == percentile
        ]

        if len(matches) != 1:
            raise ValueError("Missing or duplicate DBSCAN configuration")

        selected.append(matches[0])

    return selected


def plot_dbscan_pca(
    data: np.ndarray,
    configurations: list[dict],
    output_path: Path,
) -> dict:
    """compares DBSCAN noise assignments in shared PCA projection"""
    matrix = np.asarray(data, dtype=float)

    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("PCA requires at least two feature columns")

    if not np.isfinite(matrix).all():
        raise ValueError("PCA input contains non-finite values")

    if len(configurations) != len(COMPARISON_SETTINGS):
        raise ValueError("Expected two DBSCAN configurations")

    clustering_results = []

    for configuration in configurations:
        labels, summary = run_dbscan(
            matrix,
            eps=configuration["eps"],
            min_samples=configuration["min_samples"],
        )

        if (
            summary["noise_samples"] != configuration["noise_samples"]
            or summary["cluster_count"] != configuration["cluster_count"]
        ):
            raise ValueError("DBSCAN results differ from the saved sweep")

        clustering_results.append((labels, summary))

    pca = PCA(n_components=2, svd_solver="full")
    projected = pca.fit_transform(matrix)

    explained_variance = pca.explained_variance_ratio_

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharex=True,
        sharey=True,
    )

    for ax, configuration, result in zip(
        axes,
        configurations,
        clustering_results,
        strict=True,
    ):
        labels, summary = result

        noise_mask = labels == NOISE_LABEL
        clustered_mask = ~noise_mask

        ax.scatter(
            projected[clustered_mask, 0],
            projected[clustered_mask, 1],
            s=12,
            alpha=0.4,
            label="Clustered",
        )

        ax.scatter(
            projected[noise_mask, 0],
            projected[noise_mask, 1],
            s=15,
            alpha=0.65,
            label="Noise",
        )

        ax.set_title(
            f"eps = {configuration['eps']:.3f}, "
            f"min_samples = {configuration['min_samples']}\n"
            f"Noise: {summary['noise_fraction']:.1%}"
        )

        ax.set_xlabel(f"PC1 ({explained_variance[0]:.1%} variance)")

        ax.grid(alpha=0.2)
        ax.legend()

    axes[0].set_ylabel(f"PC2 ({explained_variance[1]:.1%} variance)")

    fig.suptitle("DBSCAN Cluster and Noise Assignments")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    return {
        "pca_explained_variance": [float(value) for value in explained_variance],
        "total_explained_variance": float(explained_variance.sum()),
        "configurations": [
            {
                "eps": float(configuration["eps"]),
                "min_samples": int(configuration["min_samples"]),
                "cluster_count": int(summary["cluster_count"]),
                "noise_samples": int(summary["noise_samples"]),
                "noise_fraction": float(summary["noise_fraction"]),
            }
            for configuration, (_, summary) in zip(
                configurations,
                clustering_results,
                strict=True,
            )
        ],
    }


def main() -> None:
    """visualizes representative DBSCAN configs."""
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    data, selected_positions, _ = prepare_clustering_data(
        features,
        samples,
    )

    with DBSCAN_SWEEP_PATH.open(encoding="utf-8") as file:
        sweep = json.load(file)

    if (
        sweep["sample_count"] != len(selected_positions)
        or sweep["feature_count"] != data.shape[1]
    ):
        raise ValueError("DBSCAN sweep does not match clustering data")

    configurations = select_dbscan_configurations(sweep)

    summary = plot_dbscan_pca(
        data,
        configurations,
        DBSCAN_FIGURE_PATH,
    )

    summary["samples"] = len(selected_positions)
    summary["holdout_used"] = False
    summary["purpose"] = "visualization_only"

    DBSCAN_VISUALIZATION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    DBSCAN_VISUALIZATION_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("DBSCAN PCA Visualization")
    print("------------------------")

    print(f"PCA explained variance: " f"{summary['total_explained_variance']:.2%}")

    for result in summary["configurations"]:
        print(
            f"eps={result['eps']:.4f}, "
            f"clusters={result['cluster_count']}, "
            f"noise={result['noise_fraction']:.2%}"
        )

    print(f"\nFigure saved to: {DBSCAN_FIGURE_PATH}")
    print(f"Summary saved to: {DBSCAN_VISUALIZATION_PATH}")


if __name__ == "__main__":
    main()