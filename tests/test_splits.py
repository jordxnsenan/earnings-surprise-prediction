import pandas as pd
import pytest

from earnings_surprise.evaluation.splits import (
    create_chronological_splits,
)


def make_sample_data() -> pd.DataFrame:
    """Create unsorted prediction samples spanning development and holdout."""
    return pd.DataFrame({
        "ticker": ["A"] * 6,
        "target_date_pub": pd.to_datetime([
            "2024-03-01",
            "2022-05-01",
            "2019-10-01",
            "2023-06-01",
            "2020-04-01",
            "2021-02-01",
        ]),
        "target_beat": [1, 0, 1, 0, 1, 0],
    })


def test_create_chronological_splits():
    samples = make_sample_data()
    original_samples = samples.copy(deep=True)

    folds, holdout_indices = create_chronological_splits(samples)

    assert len(folds) == 4
    assert len(holdout_indices) == 1

    assert samples.iloc[holdout_indices[0]]["target_date_pub"] == (
        pd.Timestamp("2024-03-01")
    )

    expected_training_counts = [1, 2, 3, 4]

    for fold, expected_count in zip(folds, expected_training_counts):
        validation_year = fold["validation_year"]
        training_indices = fold["training_indices"]
        validation_indices = fold["validation_indices"]

        training_dates = samples.iloc[
            training_indices
        ]["target_date_pub"]

        validation_dates = samples.iloc[
            validation_indices
        ]["target_date_pub"]

        assert len(training_indices) == expected_count
        assert len(validation_indices) == 1

        assert (training_dates < validation_dates.min()).all()
        assert validation_dates.dt.year.eq(validation_year).all()

        # Holdout rows must be absent from both groups.
        assert not set(holdout_indices).intersection(
            set(training_indices) | set(validation_indices)
        )

    pd.testing.assert_frame_equal(samples, original_samples)


def test_splits_reject_invalid_targets():
    samples = make_sample_data()
    samples.loc[1, "target_beat"] = 2

    with pytest.raises(ValueError, match="valid binary targets"):
        create_chronological_splits(samples)


def test_splits_reject_duplicate_prediction_dates():
    samples = make_sample_data()

    duplicate = samples.iloc[[0]].copy()
    samples = pd.concat([samples, duplicate], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate prediction samples"):
        create_chronological_splits(samples)


def test_splits_require_holdout_samples():
    samples = make_sample_data()
    samples = samples.loc[
        samples["target_date_pub"].dt.year < 2024
    ]

    with pytest.raises(ValueError, match="No final holdout samples"):
        create_chronological_splits(samples)