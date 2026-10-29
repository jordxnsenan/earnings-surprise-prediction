import numpy as np
import pandas as pd


VALIDATION_YEARS = (2020, 2021, 2022, 2023)
HOLDOUT_START = pd.Timestamp("2024-01-01")

SAMPLE_KEY_COLUMNS = ["ticker", "target_date_pub"]


def create_chronological_splits(
    samples: pd.DataFrame,
) -> tuple[list[dict], np.ndarray]:
    """creates expanding validation folds and a reserved final holdout"""
    required_columns = SAMPLE_KEY_COLUMNS + ["target_beat"]

    missing_columns = [
        column for column in required_columns
        if column not in samples.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

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

    candidate_keys = pd.DataFrame({
        "ticker": samples["ticker"].to_numpy(),
        "target_day": target_days.to_numpy(),
    })

    if candidate_keys.duplicated().any():
        raise ValueError("Duplicate prediction samples on the same day")

    # indices refer to row positions in the original samples DataFrame
    holdout_mask = target_days >= HOLDOUT_START
    holdout_indices = np.flatnonzero(holdout_mask.to_numpy())

    if len(holdout_indices) == 0:
        raise ValueError("No final holdout samples found")

    folds = []

    for validation_year in VALIDATION_YEARS:
        validation_start = pd.Timestamp(
            year=validation_year, month=1, day=1
        )

        validation_end = pd.Timestamp(
            year=validation_year + 1, month=1, day=1
        )

        training_mask = target_days < validation_start

        validation_mask = (
            (target_days >= validation_start)
            & (target_days < validation_end)
        )

        training_indices = np.flatnonzero(
            training_mask.to_numpy()
        )

        validation_indices = np.flatnonzero(
            validation_mask.to_numpy()
        )

        if len(training_indices) == 0:
            raise ValueError(
                f"No training samples for {validation_year}"
            )

        if len(validation_indices) == 0:
            raise ValueError(
                f"No validation samples for {validation_year}"
            )

        if (
            holdout_mask.iloc[training_indices].any()
            or holdout_mask.iloc[validation_indices].any()
        ):
            raise ValueError("Final holdout entered a development fold")

        folds.append({
            "validation_year": validation_year,
            "training_indices": training_indices,
            "validation_indices": validation_indices,
        })

    return folds, holdout_indices