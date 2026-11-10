import json
import numpy as np
import pandas as pd

from pathlib import Path
from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.splits import (
    HOLDOUT_START,
    VALIDATION_YEARS,
    create_chronological_splits,
)

ASSESSMENT_YEAR = 2023
SEARCH_VALIDATION_YEARS = tuple(
    year for year in VALIDATION_YEARS if year < ASSESSMENT_YEAR
)

TUNING_SPLITS_PATH = Path("results/tuning/chronological_splits.json")


def validate_positions(
    positions: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    """validates unique, positional sample indices"""
    positions = np.asarray(positions)

    if (
        positions.ndim != 1
        or not np.issubdtype(positions.dtype, np.integer)
        or len(positions) == 0
        or (positions < 0).any()
        or (positions >= sample_count).any()
        or len(np.unique(positions)) != len(positions)
    ):
        raise ValueError("Invalid chronological sample positions")

    return positions


def build_tuning_partitions(samples: pd.DataFrame) -> dict:
    """separates chronological search folds from 2023 assessment data"""
    if "target_date_pub" not in samples.columns:
        raise ValueError("Missing target_date_pub column")

    dates = pd.to_datetime(
        samples["target_date_pub"],
        errors="raise",
    )

    if dates.isna().any():
        raise ValueError("Prediction dates cannot be missing")

    folds, holdout_indices = create_chronological_splits(samples)

    expected_years = set(SEARCH_VALIDATION_YEARS) | {ASSESSMENT_YEAR}

    years = [int(fold["validation_year"]) for fold in folds]

    if len(years) != len(expected_years) or set(years) != expected_years:
        raise ValueError("Unexpected chronological validation years")

    folds_by_year = {int(fold["validation_year"]): fold for fold in folds}

    assessment_fold = folds_by_year[ASSESSMENT_YEAR]

    assessment_training = validate_positions(
        assessment_fold["training_indices"],
        len(samples),
    )

    assessment_validation = validate_positions(
        assessment_fold["validation_indices"],
        len(samples),
    )

    holdout_positions = validate_positions(
        holdout_indices,
        len(samples),
    )

    assessment_start = pd.Timestamp(f"{ASSESSMENT_YEAR}-01-01")

    if (
        (dates.iloc[assessment_training] >= assessment_start).any()
        or not (dates.iloc[assessment_validation].dt.year == ASSESSMENT_YEAR).all()
        or (dates.iloc[assessment_validation] >= HOLDOUT_START).any()
        or (dates.iloc[holdout_positions] < HOLDOUT_START).any()
    ):
        raise ValueError("Invalid assessment or holdout dates")

    if (
        np.intersect1d(
            assessment_training,
            assessment_validation,
        ).size
        or np.intersect1d(
            assessment_training,
            holdout_positions,
        ).size
        or np.intersect1d(
            assessment_validation,
            holdout_positions,
        ).size
    ):
        raise ValueError("Assessment and holdout positions overlap")

    # these become the rows passed to RandomizedSearchCV
    search_positions = np.sort(assessment_training)

    position_lookup = np.full(
        len(samples),
        fill_value=-1,
        dtype=int,
    )

    position_lookup[search_positions] = np.arange(len(search_positions))

    search_folds = []

    for year in SEARCH_VALIDATION_YEARS:
        fold = folds_by_year[year]

        original_training = validate_positions(
            fold["training_indices"],
            len(samples),
        )

        original_validation = validate_positions(
            fold["validation_indices"],
            len(samples),
        )

        training_dates = dates.iloc[original_training]
        validation_dates = dates.iloc[original_validation]

        if (
            np.intersect1d(
                original_training,
                original_validation,
            ).size
            or training_dates.max() >= validation_dates.min()
            or not (validation_dates.dt.year == year).all()
        ):
            raise ValueError("Invalid chronological search fold")

        local_training = position_lookup[original_training]
        local_validation = position_lookup[original_validation]

        if (local_training < 0).any() or (local_validation < 0).any():
            raise ValueError(
                "Search folds contain observations outside the search period"
            )

        search_folds.append(
            {
                "validation_year": year,
                "training_indices": local_training,
                "validation_indices": local_validation,
            }
        )

    return {
        "search_positions": search_positions,
        "search_folds": search_folds,
        "assessment_training_positions": assessment_training,
        "assessment_validation_positions": assessment_validation,
        "holdout_positions": holdout_positions,
    }


def main() -> None:
    """saves the verified chronological tuning arrangement"""
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    partitions = build_tuning_partitions(samples)

    summary = {
        "experiment": "chronological_hyperparameter_tuning_splits",
        "search_validation_years": list(SEARCH_VALIDATION_YEARS),
        "assessment_year": ASSESSMENT_YEAR,
        "search_samples": len(partitions["search_positions"]),
        "assessment_training_samples": len(partitions["assessment_training_positions"]),
        "assessment_validation_samples": len(
            partitions["assessment_validation_positions"]
        ),
        "holdout_samples": len(partitions["holdout_positions"]),
        "holdout_used": False,
        "search_folds": [
            {
                "validation_year": fold["validation_year"],
                "training_samples": len(fold["training_indices"]),
                "validation_samples": len(fold["validation_indices"]),
            }
            for fold in partitions["search_folds"]
        ],
    }

    TUNING_SPLITS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    TUNING_SPLITS_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Chronological Hyperparameter Tuning Splits")
    print("-----------------------------------------")
    print(f"Search observations: {summary['search_samples']:,}")

    for fold in summary["search_folds"]:
        print(
            f"{fold['validation_year']}: "
            f"{fold['training_samples']:,} training, "
            f"{fold['validation_samples']:,} validation"
        )

    print(
        f"2023 assessment: "
        f"{summary['assessment_validation_samples']:,} observations"
    )

    print(f"2024 reserved holdout: " f"{summary['holdout_samples']:,} observations")

    print(f"\nResults saved to: {TUNING_SPLITS_PATH}")


if __name__ == "__main__":
    main()
