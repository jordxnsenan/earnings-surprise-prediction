import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.decomposition import PCA

from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.features import FEATURE_MATRIX_PATH
from earnings_surprise.outliers.isolation_forest import (
    ISOLATION_OUTLIER_LABEL,
    ISOLATION_RESULTS_PATH,
    run_isolation_forest,
)
from earnings_surprise.outliers.prepare import (
    OUTLIER_RANDOM_STATE,
    prepare_outlier_data,
)

PROFILE_COLUMNS = (
    "actual_eps_lag1",
    "estimate_eps_lag1",
    "surprise_mean4",
    "surprise_std4",
    "beat_rate4",
    "beat_streak4",
    "eps_change1_2",
    "eps_trend4",
)

PCA_SAMPLE_SIZE = 4000
HISTOGRAM_BINS = 50
FIGURE_DPI = 200
SCORE_COMPARISON_TOLERANCE = 1e-10

PROFILE_PATH = Path("results/outliers/isolation_forest_profile.csv")
ASSIGNMENTS_PATH = Path("results/outliers/isolation_forest_assignments.parquet")
FIGURE_PATH = Path("results/figures/isolation_forest_diagnostics.png")
ANALYSIS_PATH = Path("results/outliers/isolation_forest_analysis.json")


def summarize_outlier_features(
    features: pd.DataFrame,
    positions: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """compares original historical features for inliers and outliers"""
    positions = np.asarray(positions)
    labels = np.asarray(labels)

    if len(positions) != len(labels):
        raise ValueError("Positions and labels must have equal lengths")

    if (
        positions.ndim != 1
        or not np.issubdtype(positions.dtype, np.integer)
        or np.any(positions < 0)
        or np.any(positions >= len(features))
        or len(np.unique(positions)) != len(positions)
    ):
        raise ValueError("Invalid source row positions")

    if not np.isin(labels, [-1, 1]).all():
        raise ValueError("Invalid Isolation Forest labels")

    if not np.any(labels == -1) or not np.any(labels == 1):
        raise ValueError("Both inliers and outliers are required")

    missing_columns = set(PROFILE_COLUMNS) - set(features.columns)

    if missing_columns:
        raise ValueError("Missing profile features")

    selected = features.iloc[positions]
    inliers = selected.iloc[np.flatnonzero(labels == 1)]
    outliers = selected.iloc[np.flatnonzero(labels == -1)]

    rows = []

    for feature in PROFILE_COLUMNS:
        inlier_values = pd.to_numeric(inliers[feature], errors="coerce")
        outlier_values = pd.to_numeric(outliers[feature], errors="coerce")

        rows.append(
            {
                "feature": feature,
                "inlier_median": float(inlier_values.median()),
                "outlier_median": float(outlier_values.median()),
                "inlier_median_absolute": float(inlier_values.abs().median()),
                "outlier_median_absolute": float(outlier_values.abs().median()),
                "inlier_missing_fraction": float(inlier_values.isna().mean()),
                "outlier_missing_fraction": float(outlier_values.isna().mean()),
            }
        )

    return pd.DataFrame(rows)


def plot_isolation_diagnostics(
    data: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
) -> list[float]:
    """plots anomaly scores and sampled observations in PCA space"""
    matrix = np.asarray(data, dtype=float)
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)

    if (
        matrix.ndim != 2
        or matrix.shape[0] < 2
        or matrix.shape[1] < 2
        or len(matrix) != len(labels)
        or len(matrix) != len(scores)
    ):
        raise ValueError("Invalid diagnostic data dimensions")

    if not np.isfinite(matrix).all() or not np.isfinite(scores).all():
        raise ValueError("Diagnostic data must be finite")

    if not np.isin(labels, [-1, 1]).all():
        raise ValueError("Invalid anomaly labels")

    pca = PCA(n_components=2, svd_solver="full")
    projection = pca.fit_transform(matrix)

    generator = np.random.default_rng(OUTLIER_RANDOM_STATE)
    selected = np.sort(
        generator.choice(
            len(matrix),
            size=min(PCA_SAMPLE_SIZE, len(matrix)),
            replace=False,
        )
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    groups = (
        (1, "Inliers"),
        (ISOLATION_OUTLIER_LABEL, "Outliers"),
    )

    for label, name in groups:
        group_mask = labels == label

        axes[0].hist(
            scores[group_mask],
            bins=HISTOGRAM_BINS,
            alpha=0.6,
            label=f"{name} (n={group_mask.sum():,})",
        )

        plot_mask = labels[selected] == label
        plot_positions = selected[plot_mask]

        axes[1].scatter(
            projection[plot_positions, 0],
            projection[plot_positions, 1],
            s=12,
            alpha=0.5,
            label=name,
        )

    axes[0].set_title("Isolation Forest anomaly scores")
    axes[0].set_xlabel("Score (lower = more anomalous)")
    axes[0].set_ylabel("Observations")
    axes[0].legend()

    axes[1].set_title(f"PCA projection ({len(selected):,} sampled observations)")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    axes[1].legend()

    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)

    return [float(value) for value in pca.explained_variance_ratio_]


def main() -> None:
    """Profile the detected anomalies in the development data."""
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    data, positions, _ = prepare_outlier_data(features, samples)

    labels, scores, summary = run_isolation_forest(data)

    with ISOLATION_RESULTS_PATH.open(encoding="utf-8") as file:
        previous = json.load(file)

    if (
        previous["samples"] != summary["samples"]
        or previous["outlier_count"] != summary["outlier_count"]
    ):
        raise ValueError("Isolation Forest results changed")

    for percentile, previous_score in previous["score_percentiles"].items():
        current_score = summary["score_percentiles"][percentile]

        if not np.isclose(
            previous_score,
            current_score,
            rtol=0,
            atol=SCORE_COMPARISON_TOLERANCE,
        ):
            raise ValueError("Isolation Forest anomaly scores changed")

    profile = summarize_outlier_features(features, positions, labels)

    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(PROFILE_PATH, index=False)

    assignments = pd.DataFrame(
        {
            "source_row_position": positions,
            "anomaly_label": labels,
            "anomaly_score": scores,
        }
    )

    assignments.to_parquet(ASSIGNMENTS_PATH, index=False)

    explained_variance = plot_isolation_diagnostics(data, labels, scores, FIGURE_PATH)

    report = {
        "experiment": "isolation_forest_anomaly_analysis",
        "samples": len(positions),
        "outlier_count": summary["outlier_count"],
        "outlier_fraction": summary["outlier_fraction"],
        "holdout_used": False,
        "pca_explained_variance_ratio": explained_variance,
        "pca_total_explained_variance": sum(explained_variance),
        "pca_plot_sample_size": min(PCA_SAMPLE_SIZE, len(data)),
        "note": (
            "Exploratory development analysis only. "
            "Refit detectors within training folds before "
            "evaluating outlier removal."
        ),
    }

    ANALYSIS_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Isolation Forest — Anomaly Analysis")
    print("-----------------------------------")
    print(f"Outliers: {summary['outlier_count']:,}")
    print("PCA explained variance: " f"{report['pca_total_explained_variance']:.2%}")

    print("\nOriginal feature profiles:")
    print(profile.round(4).to_string(index=False))

    print(f"\nProfile saved to: {PROFILE_PATH}")
    print(f"Assignments saved to: {ASSIGNMENTS_PATH}")
    print(f"Figure saved to: {FIGURE_PATH}")
    print(f"Analysis saved to: {ANALYSIS_PATH}")


if __name__ == "__main__":
    main()
