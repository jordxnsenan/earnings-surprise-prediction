import numpy as np
import pandas as pd
import json

from pathlib import Path
from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH

VALIDATION_YEARS = (2020, 2021, 2022, 2023)
HOLDOUT_START = pd.Timestamp("2024-01-01")

SAMPLE_KEY_COLUMNS = ["ticker", "target_date_pub"]


def create_chronological_splits(
    samples: pd.DataFrame,
) -> tuple[list[dict], np.ndarray]:
    """creates expanding validation folds and a reserved final holdout"""
    required_columns = SAMPLE_KEY_COLUMNS + ["target_beat"]

    missing_columns = [
        column for column in required_columns if column not in samples.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if samples.empty:
        raise ValueError("No prediction samples available")

    target_dates = pd.to_datetime(
        samples["target_date_pub"],
        errors="raise",
    )

    if samples["ticker"].isna().any() or target_dates.isna().any():
        raise ValueError("Prediction sample identifiers cannot be missing")

    if not samples["target_beat"].isin([0, 1]).all():
        raise ValueError("Prediction samples must have valid binary targets")

    # works w/ calendar dates due to intraday ordering being unverified
    target_days = target_dates.dt.normalize()

    candidate_keys = pd.DataFrame(
        {
            "ticker": samples["ticker"].to_numpy(),
            "target_day": target_days.to_numpy(),
        }
    )

    if candidate_keys.duplicated().any():
        raise ValueError("Duplicate prediction samples on the same day")

    # indices refer to row positions in the original samples DataFrame
    holdout_mask = target_days >= HOLDOUT_START
    holdout_indices = np.flatnonzero(holdout_mask.to_numpy())

    if len(holdout_indices) == 0:
        raise ValueError("No final holdout samples found")

    folds = []

    for validation_year in VALIDATION_YEARS:
        validation_start = pd.Timestamp(year=validation_year, month=1, day=1)

        validation_end = pd.Timestamp(year=validation_year + 1, month=1, day=1)

        training_mask = target_days < validation_start

        validation_mask = (target_days >= validation_start) & (
            target_days < validation_end
        )

        training_indices = np.flatnonzero(training_mask.to_numpy())

        validation_indices = np.flatnonzero(validation_mask.to_numpy())

        if len(training_indices) == 0:
            raise ValueError(f"No training samples for {validation_year}")

        if len(validation_indices) == 0:
            raise ValueError(f"No validation samples for {validation_year}")

        if (
            holdout_mask.iloc[training_indices].any()
            or holdout_mask.iloc[validation_indices].any()
        ):
            raise ValueError("Final holdout entered a development fold")

        folds.append(
            {
                "validation_year": validation_year,
                "training_indices": training_indices,
                "validation_indices": validation_indices,
            }
        )

    return folds, holdout_indices


SPLIT_SUMMARY_PATH = Path("results/split_coverage.json")


def summarize_class_balance(
    samples: pd.DataFrame,
    indices: np.ndarray,
) -> dict:
    """summarizes earnings outcomes for a development subset"""
    labels = samples.iloc[indices]["target_beat"]

    beat_count = int(labels.eq(1).sum())
    meet_or_miss_count = int(labels.eq(0).sum())

    return {
        "beats": beat_count,
        "meet_or_miss": meet_or_miss_count,
        "beat_rate": round(beat_count / len(labels), 4),
    }


def summarize_split_coverage(samples: pd.DataFrame) -> dict:
    """summarizes chronological folds without examining holdout labels"""
    folds, holdout_indices = create_chronological_splits(samples)

    target_dates = pd.to_datetime(samples["target_date_pub"]).dt.normalize()

    holdout_dates = target_dates.iloc[holdout_indices]

    summary = {
        "total_samples": len(samples),
        
        "development_samples": len(samples) - len(holdout_indices),
        
        "holdout": {
            "samples": len(holdout_indices),
            "first_date": holdout_dates.min().date().isoformat(),
            "last_date": holdout_dates.max().date().isoformat(),
        },
        
        "folds": [],
        
        "evaluation_policy": {
            "strategy": "expanding_calendar_year",
            "date_column": "target_date_pub",
            "validation_years": list(VALIDATION_YEARS),
            "holdout_start": HOLDOUT_START.date().isoformat(),
            "primary_selection_metric": "macro_f1",
            "availability_basis": "provisional_publication_dates",
        },
    }

    for fold in folds:
        year = fold["validation_year"]
        training_indices = fold["training_indices"]
        validation_indices = fold["validation_indices"]

        training_dates = target_dates.iloc[training_indices]
        validation_dates = target_dates.iloc[validation_indices]

        if training_dates.max() >= validation_dates.min():
            raise ValueError(f"Training dates overlap validation dates for {year}")

        fold_summary = {
            "validation_year": year,
            "training_samples": len(training_indices),
            "validation_samples": len(validation_indices),
            "last_training_date": (training_dates.max().date().isoformat()),
            "first_validation_date": (validation_dates.min().date().isoformat()),
            "last_validation_date": (validation_dates.max().date().isoformat()),
            "training_class_balance": summarize_class_balance(
                samples, training_indices
            ),
            "validation_class_balance": summarize_class_balance(
                samples, validation_indices
            ),
        }

        summary["folds"].append(fold_summary)

    return summary


def main() -> None:
    """generates reproducible chronological coverage report"""
    if not PREDICTION_WINDOWS_PATH.is_file():
        raise FileNotFoundError(
            f"Prediction samples not found: {PREDICTION_WINDOWS_PATH}"
        )

    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)
    summary = summarize_split_coverage(samples)

    SPLIT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    SPLIT_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {SPLIT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
