import numpy as np
import pandas as pd
import pytest

from earnings_surprise.tuning.splits import (
    ASSESSMENT_YEAR,
    SEARCH_VALIDATION_YEARS,
    build_tuning_partitions,
    validate_positions,
)


def make_test_samples():
    """creates historical observations for each relevant year"""
    dates = []

    for year, count in (
        (2019, 5),
        (2020, 4),
        (2021, 4),
        (2022, 4),
        (2023, 4),
        (2024, 4),
    ):
        dates.extend(
            pd.date_range(
                f"{year}-01-01",
                periods=count,
                freq="D",
            )
        )

    return pd.DataFrame(
        {
            "ticker": ["TEST"] * len(dates),
            "target_date_pub": dates,
            "target_beat": np.arange(len(dates)) % 2,
        }
    )


def test_tuning_splits_separate_search_and_assessment():
    samples = make_test_samples()

    partitions = build_tuning_partitions(samples)

    search_positions = partitions["search_positions"]
    assessment_positions = partitions["assessment_validation_positions"]

    dates = pd.to_datetime(samples["target_date_pub"])

    assert len(partitions["search_folds"]) == len(SEARCH_VALIDATION_YEARS)

    assert len(search_positions) == 17
    assert len(assessment_positions) == 4
    assert len(partitions["holdout_positions"]) == 4

    assert (dates.iloc[search_positions].dt.year < 2023).all()
    assert (dates.iloc[assessment_positions].dt.year == ASSESSMENT_YEAR).all()


def test_search_fold_indices_map_to_search_dataset():
    samples = make_test_samples()

    partitions = build_tuning_partitions(samples)
    search_positions = partitions["search_positions"]

    for fold in partitions["search_folds"]:
        training = fold["training_indices"]
        validation = fold["validation_indices"]

        assert (training >= 0).all()
        assert (validation >= 0).all()

        assert (training < len(search_positions)).all()
        assert (validation < len(search_positions)).all()

        assert np.intersect1d(training, validation).size == 0

        original_validation = search_positions[validation]

        validation_years = pd.to_datetime(
            samples.iloc[original_validation]["target_date_pub"]
        ).dt.year

        assert (validation_years == fold["validation_year"]).all()


def test_tuning_splits_work_with_shuffled_rows():
    samples = (
        make_test_samples()
        .sample(
            frac=1,
            random_state=42,
        )
        .reset_index(drop=True)
    )

    partitions = build_tuning_partitions(samples)

    assert len(partitions["search_positions"]) == 17
    assert len(partitions["search_folds"]) == 3

    dates = pd.to_datetime(samples["target_date_pub"])

    search_dates = dates.iloc[partitions["search_positions"]]

    assert (search_dates.dt.year < 2023).all()


def test_tuning_splits_ignore_target_values():
    samples = make_test_samples()

    original = build_tuning_partitions(samples)

    changed_samples = samples.copy()
    changed_samples["target_beat"] = 1

    modified = build_tuning_partitions(changed_samples)

    np.testing.assert_array_equal(
        original["search_positions"],
        modified["search_positions"],
    )

    for first, second in zip(
        original["search_folds"],
        modified["search_folds"],
        strict=True,
    ):
        np.testing.assert_array_equal(
            first["training_indices"],
            second["training_indices"],
        )

        np.testing.assert_array_equal(
            first["validation_indices"],
            second["validation_indices"],
        )


def test_validate_positions_rejects_duplicates():
    with pytest.raises(ValueError, match="Invalid chronological"):
        validate_positions(np.array([0, 0, 2]), 5)


def test_validate_positions_rejects_out_of_range():
    with pytest.raises(ValueError, match="Invalid chronological"):
        validate_positions(np.array([0, 1, 5]), 5)


def test_tuning_splits_reject_missing_dates():
    samples = make_test_samples()

    samples.loc[0, "target_date_pub"] = pd.NaT

    with pytest.raises(ValueError, match="cannot be missing"):
        build_tuning_partitions(samples)
