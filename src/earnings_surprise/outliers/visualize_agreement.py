import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.decomposition import PCA

from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.splits import HOLDOUT_START
from earnings_surprise.features import FEATURE_MATRIX_PATH
from earnings_surprise.outliers.analyze_isolation import PROFILE_COLUMNS
from earnings_surprise.outliers.compare_all_detectors import (
    ALL_DETECTORS_ASSIGNMENTS_PATH,
    ALL_DETECTORS_RESULTS_PATH,
    summarize_three_detector_agreement,
)
from earnings_surprise.outliers.prepare import (
    OUTLIER_COMPARISON_SAMPLE_SIZE,
    prepare_outlier_data,
)

DETECTOR_LABEL_COLUMNS = (
    "isolation_label",
    "lof_label",
    "svm_label",
)

AGREEMENT_GROUPS = (
    "none",
    "exactly_one",
    "exactly_two",
    "all_three",
)

AGREEMENT_LABELS = (
    "No detectors",
    "One detector",
    "Two detectors",
    "All three detectors",
)

AGREEMENT_COLORS = (
    "#B0B7C1",
    "#4C91C7",
    "#E6A04A",
    "#C44857",
)

OUTLIER_LABEL = -1
FIGURE_DPI = 200

AGREEMENT_PROFILE_PATH = Path("results/outliers/three_detector_profiles.csv")

AGREEMENT_FIGURE_PATH = Path("results/figures/three_detector_pca_agreement.png")

AGREEMENT_ANALYSIS_PATH = Path("results/outliers/three_detector_analysis.json")


def validate_agreement_assignments(
    assignments: pd.DataFrame,
) -> np.ndarray:
    """verifies that saved agreement groups match detector labels"""
    required_columns = set(DETECTOR_LABEL_COLUMNS) | {
        "detector_count",
        "agreement_group",
    }

    if assignments.empty:
        raise ValueError("Anomaly assignments cannot be empty")

    if not required_columns.issubset(assignments.columns):
        raise ValueError("Missing detector agreement columns")

    labels = assignments[list(DETECTOR_LABEL_COLUMNS)].to_numpy()

    if not np.isin(labels, [-1, 1]).all():
        raise ValueError("Invalid anomaly labels")

    detector_counts = np.sum(
        labels == OUTLIER_LABEL,
        axis=1,
    )

    saved_counts = pd.to_numeric(
        assignments["detector_count"],
        errors="coerce",
    ).to_numpy()

    if not np.array_equal(detector_counts, saved_counts):
        raise ValueError("Detector counts do not match labels")

    expected_groups = np.asarray(AGREEMENT_GROUPS)[detector_counts]

    if not np.array_equal(
        assignments["agreement_group"].to_numpy(),
        expected_groups,
    ):
        raise ValueError("Agreement groups do not match labels")

    return expected_groups


def summarize_agreement_profiles(
    features: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """profiles anomaly agreement groups using original features"""
    groups = validate_agreement_assignments(assignments)

    if "source_row_position" not in assignments.columns:
        raise ValueError("Missing source row positions")

    positions = assignments["source_row_position"].to_numpy()

    if (
        not np.issubdtype(positions.dtype, np.integer)
        or (positions < 0).any()
        or (positions >= len(features)).any()
        or len(np.unique(positions)) != len(positions)
    ):
        raise ValueError("Invalid source row positions")

    if not set(PROFILE_COLUMNS).issubset(features.columns):
        raise ValueError("Missing historical profile features")

    selected_features = features.iloc[positions].reset_index(drop=True)

    rows = []

    for group_name in AGREEMENT_GROUPS:
        group_mask = groups == group_name
        group_count = int(group_mask.sum())

        if group_count == 0:
            continue

        group_features = selected_features.iloc[np.flatnonzero(group_mask)]

        for feature in PROFILE_COLUMNS:
            values = pd.to_numeric(
                group_features[feature],
                errors="coerce",
            ).replace(
                [np.inf, -np.inf],
                np.nan,
            )

            rows.append(
                {
                    "agreement_group": group_name,
                    "samples": group_count,
                    "feature": feature,
                    "median": float(values.median()),
                    "median_absolute": float(values.abs().median()),
                    "missing_fraction": float(values.isna().mean()),
                }
            )

    return pd.DataFrame(rows)


def plot_agreement_pca(
    data: np.ndarray,
    assignments: pd.DataFrame,
    output_path: Path,
) -> list[float]:
    """visualizes detector agreement in two PCA dimensions"""
    matrix = np.asarray(data, dtype=float)
    groups = validate_agreement_assignments(assignments)

    if (
        matrix.ndim != 2
        or matrix.shape[0] < 2
        or matrix.shape[1] < 2
        or len(matrix) != len(assignments)
    ):
        raise ValueError("Invalid PCA data dimensions")

    if not np.isfinite(matrix).all():
        raise ValueError("PCA data must contain finite values")

    pca = PCA(n_components=2, svd_solver="full")
    projection = pca.fit_transform(matrix)

    figure, ax = plt.subplots(figsize=(10, 7))

    for group_name, group_label, color in zip(
        AGREEMENT_GROUPS,
        AGREEMENT_LABELS,
        AGREEMENT_COLORS,
        strict=True,
    ):
        group_mask = groups == group_name
        group_count = int(group_mask.sum())

        if group_count == 0:
            continue

        ax.scatter(
            projection[group_mask, 0],
            projection[group_mask, 1],
            s=12 if group_name == "none" else 22,
            alpha=0.25 if group_name == "none" else 0.75,
            color=color,
            label=f"{group_label} (n={group_count:,})",
        )

    explained_variance = pca.explained_variance_ratio_

    ax.set_title("Agreement Between Three Outlier Detectors")
    ax.set_xlabel(f"PC1 ({explained_variance[0]:.1%} variance explained)")
    ax.set_ylabel(f"PC2 ({explained_variance[1]:.1%} variance explained)")

    ax.grid(alpha=0.2)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)

    return [float(value) for value in explained_variance]


def main() -> None:
    """visualizes agreement and compares original earnings features"""
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    data, selected_positions, _ = prepare_outlier_data(
        features,
        samples,
        sample_size=OUTLIER_COMPARISON_SAMPLE_SIZE,
    )

    assignments = pd.read_csv(ALL_DETECTORS_ASSIGNMENTS_PATH)

    if not np.array_equal(
        assignments["source_row_position"].to_numpy(),
        selected_positions,
    ):
        raise ValueError("Saved assignments do not match sampled rows")

    selected_dates = pd.to_datetime(samples.iloc[selected_positions]["target_date_pub"])

    if selected_dates.isna().any() or (selected_dates >= HOLDOUT_START).any():
        raise ValueError("Holdout observations entered the analysis")

    with ALL_DETECTORS_RESULTS_PATH.open(encoding="utf-8") as file:
        previous_report = json.load(file)

    if (
        previous_report["sample_count"] != len(data)
        or previous_report["feature_count"] != data.shape[1]
        or previous_report["holdout_used"] is not False
    ):
        raise ValueError("Saved comparison metadata does not match")

    agreement = summarize_three_detector_agreement(assignments)

    if (
        agreement["agreement_counts"] != previous_report["agreement_counts"]
        or agreement["detector_outliers"] != previous_report["detector_outliers"]
    ):
        raise ValueError("Saved anomaly agreement counts changed")

    profiles = summarize_agreement_profiles(
        features,
        assignments,
    )

    explained_variance = plot_agreement_pca(
        data,
        assignments,
        AGREEMENT_FIGURE_PATH,
    )

    AGREEMENT_PROFILE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profiles.to_csv(
        AGREEMENT_PROFILE_PATH,
        index=False,
    )

    report = {
        "experiment": "three_detector_agreement_analysis",
        "sample_count": len(data),
        "feature_count": data.shape[1],
        "holdout_used": False,
        "agreement_counts": agreement["agreement_counts"],
        "profile_features": list(PROFILE_COLUMNS),
        "pca_explained_variance_ratio": explained_variance,
        "pca_total_explained_variance": float(sum(explained_variance)),
        "pca_purpose": "visualization_only",
    }

    AGREEMENT_ANALYSIS_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Three-Detector Anomaly Analysis")
    print("-------------------------------")

    for group, count in agreement["agreement_counts"].items():
        print(f"{group}: {count:,}")

    print("\nPCA explained variance: " f"{report['pca_total_explained_variance']:.2%}")

    important_features = (
        "actual_eps_lag1",
        "surprise_mean4",
        "surprise_std4",
        "beat_rate4",
    )

    print("\nSelected historical feature profiles:")

    selected_profiles = profiles[profiles["feature"].isin(important_features)]

    print(
        selected_profiles[
            [
                "agreement_group",
                "feature",
                "median",
                "median_absolute",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print(f"\nProfile saved to: {AGREEMENT_PROFILE_PATH}")
    print(f"Figure saved to: {AGREEMENT_FIGURE_PATH}")
    print(f"Analysis saved to: {AGREEMENT_ANALYSIS_PATH}")


if __name__ == "__main__":
    main()
