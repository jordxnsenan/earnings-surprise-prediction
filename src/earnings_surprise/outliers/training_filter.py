import numpy as np
import pandas as pd

from earnings_surprise.evaluation.splits import HOLDOUT_START
from earnings_surprise.outliers.isolation_forest import (
    ISOLATION_CONTAMINATION,
    ISOLATION_TREE_COUNT,
    run_isolation_forest,
)
from earnings_surprise.outliers.prepare import prepare_outlier_data

INLIER_LABEL = 1


def validate_fold_positions(
    positions: np.ndarray,
    row_count: int,
) -> np.ndarray:
    """validates positional indices for a chronological fold"""
    positions = np.asarray(positions)

    if (
        positions.ndim != 1
        or not np.issubdtype(positions.dtype, np.integer)
        or len(positions) == 0
        or (positions < 0).any()
        or (positions >= row_count).any()
        or len(np.unique(positions)) != len(positions)
    ):
        raise ValueError("Invalid chronological fold positions")

    return positions


def filter_training_fold(
    features: pd.DataFrame,
    samples: pd.DataFrame,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
    contamination: float = ISOLATION_CONTAMINATION,
    tree_count: int = ISOLATION_TREE_COUNT,
) -> tuple[np.ndarray, dict]:
    """removes outliers identified using only a fold's training rows"""
    if len(features) != len(samples):
        raise ValueError("Feature and sample counts do not match")

    if not features.index.equals(samples.index):
        raise ValueError("Feature and sample indices do not match")

    if "target_date_pub" not in samples.columns:
        raise ValueError("Missing target_date_pub column")

    training_positions = validate_fold_positions(
        training_indices,
        len(samples),
    )

    validation_positions = validate_fold_positions(
        validation_indices,
        len(samples),
    )

    if (
        np.intersect1d(
            training_positions,
            validation_positions,
        ).size
        > 0
    ):
        raise ValueError("Training and validation folds overlap")

    dates = pd.to_datetime(
        samples["target_date_pub"],
        errors="raise",
    )

    training_dates = dates.iloc[training_positions]
    validation_dates = dates.iloc[validation_positions]

    if training_dates.isna().any() or validation_dates.isna().any():
        raise ValueError("Fold dates cannot be missing")

    if (training_dates >= HOLDOUT_START).any() or (
        validation_dates >= HOLDOUT_START
    ).any():
        raise ValueError("Holdout observations entered a fold")

    if training_dates.max() >= validation_dates.min():
        raise ValueError("Training observations must precede validation")

    # ensures only training rows enter the fitted preprocessing pipeline
    training_features = features.iloc[training_positions].reset_index(drop=True)

    training_samples = samples.iloc[training_positions].reset_index(drop=True)

    transformed, local_positions, _ = prepare_outlier_data(
        training_features,
        training_samples,
    )

    if not np.array_equal(
        local_positions,
        np.arange(len(training_positions)),
    ):
        raise ValueError("Training preparation changed row selection")

    labels, _, detection_summary = run_isolation_forest(
        transformed,
        contamination=contamination,
        tree_count=tree_count,
    )

    retained_positions = training_positions[labels == INLIER_LABEL]

    if len(retained_positions) < 2:
        raise ValueError("Too few training observations remain")

    removed_count = len(training_positions) - len(retained_positions)

    summary = {
        "algorithm": "isolation_forest",
        "training_samples": len(training_positions),
        "retained_training_samples": len(retained_positions),
        "removed_training_samples": removed_count,
        "removed_fraction": removed_count / len(training_positions),
        "validation_samples": len(validation_positions),
        "contamination": float(contamination),
        "tree_count": int(tree_count),
        "training_last_date": (training_dates.max().date().isoformat()),
        "validation_first_date": (validation_dates.min().date().isoformat()),
        "holdout_used": False,
        "validation_filtered": False,
        "anomaly_score_median": (detection_summary["score_percentiles"]["50"]),
    }

    return retained_positions, summary
