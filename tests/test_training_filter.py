import numpy as np
import pandas as pd
import pytest

from earnings_surprise.features import FEATURE_COLUMNS
from earnings_surprise.outliers.training_filter import (
    filter_training_fold,
)
from earnings_surprise.preprocessing import CONTINUOUS_COLUMNS

TRAINING_COUNT = 24
VALIDATION_COUNT = 8
HOLDOUT_COUNT = 4
TOTAL_COUNT = TRAINING_COUNT + VALIDATION_COUNT + HOLDOUT_COUNT

TEST_TREE_COUNT = 32
TEST_CONTAMINATION = 0.10


def make_test_data():
    """creates historical features spanning training, validation and holdout"""
    positions = np.arange(TOTAL_COUNT)

    data = {
        column: np.sin(positions / (index + 2))
        for index, column in enumerate(CONTINUOUS_COLUMNS)
    }

    for lag in range(1, 5):
        data[f"beat_lag{lag}"] = (positions // (2 ** (lag - 1))) % 2

    beat_history = np.column_stack([data[f"beat_lag{lag}"] for lag in range(1, 5)])

    data["beat_rate4"] = beat_history.mean(axis=1)

    streaks = np.zeros(TOTAL_COUNT, dtype=int)
    consecutive = np.ones(TOTAL_COUNT, dtype=bool)

    for lag in range(4):
        consecutive &= beat_history[:, lag] == 1
        streaks += consecutive

    data["beat_streak4"] = streaks

    months = (positions % 12) + 1

    data["previous_report_month"] = pd.Categorical(
        months,
        categories=range(1, 13),
    )

    data["previous_report_quarter"] = pd.Categorical(
        (months - 1) // 3 + 1,
        categories=range(1, 5),
    )

    features = pd.DataFrame(data)[FEATURE_COLUMNS]

    dates = (
        list(pd.date_range("2022-01-01", periods=TRAINING_COUNT))
        + list(pd.date_range("2023-01-01", periods=VALIDATION_COUNT))
        + list(pd.date_range("2024-01-01", periods=HOLDOUT_COUNT))
    )

    samples = pd.DataFrame(
        {
            "target_date_pub": dates,
            "target_beat": positions % 2,
        }
    )

    training_indices = np.arange(TRAINING_COUNT)

    validation_indices = np.arange(
        TRAINING_COUNT,
        TRAINING_COUNT + VALIDATION_COUNT,
    )

    return features, samples, training_indices, validation_indices


def run_test_filter(features, samples, training, validation):
    """Use a smaller forest to keep unit tests efficient."""
    return filter_training_fold(
        features,
        samples,
        training,
        validation,
        contamination=TEST_CONTAMINATION,
        tree_count=TEST_TREE_COUNT,
    )


def test_training_filter_removes_only_training_rows():
    features, samples, training, validation = make_test_data()

    original_features = features.copy(deep=True)
    original_samples = samples.copy(deep=True)

    retained, summary = run_test_filter(
        features,
        samples,
        training,
        validation,
    )

    assert len(retained) < len(training)
    assert len(retained) >= 2

    assert np.isin(retained, training).all()
    assert not np.isin(retained, validation).any()

    assert summary["training_samples"] == TRAINING_COUNT
    assert summary["validation_samples"] == VALIDATION_COUNT
    assert summary["removed_training_samples"] == (TRAINING_COUNT - len(retained))

    assert summary["validation_filtered"] is False
    assert summary["holdout_used"] is False

    pd.testing.assert_frame_equal(features, original_features)
    pd.testing.assert_frame_equal(samples, original_samples)


def test_training_filter_is_reproducible():
    features, samples, training, validation = make_test_data()

    first_positions, first_summary = run_test_filter(
        features,
        samples,
        training,
        validation,
    )

    second_positions, second_summary = run_test_filter(
        features,
        samples,
        training,
        validation,
    )

    np.testing.assert_array_equal(
        first_positions,
        second_positions,
    )

    assert first_summary == second_summary


def test_validation_changes_do_not_change_training_filter():
    features, samples, training, validation = make_test_data()

    original_positions, original_summary = run_test_filter(
        features,
        samples,
        training,
        validation,
    )

    changed_features = features.copy(deep=True)
    changed_samples = samples.copy(deep=True)

    changed_features.loc[validation, "actual_eps_lag1"] = 1_000_000.0

    changed_samples.loc[validation, "target_beat"] = 1

    changed_positions, changed_summary = run_test_filter(
        changed_features,
        changed_samples,
        training,
        validation,
    )

    np.testing.assert_array_equal(
        original_positions,
        changed_positions,
    )

    assert original_summary == changed_summary


def test_training_filter_does_not_use_classification_target():
    features, samples, training, validation = make_test_data()

    first_positions, _ = run_test_filter(
        features,
        samples,
        training,
        validation,
    )

    modified_samples = samples.copy(deep=True)
    modified_samples["target_beat"] = 1

    second_positions, _ = run_test_filter(
        features,
        modified_samples,
        training,
        validation,
    )

    np.testing.assert_array_equal(
        first_positions,
        second_positions,
    )


def test_training_filter_rejects_overlapping_folds():
    features, samples, training, validation = make_test_data()

    overlapping_validation = np.concatenate(
        [
            validation,
            training[:1],
        ]
    )

    with pytest.raises(ValueError, match="overlap"):
        run_test_filter(
            features,
            samples,
            training,
            overlapping_validation,
        )


def test_training_filter_rejects_reversed_dates():
    features, samples, training, validation = make_test_data()

    with pytest.raises(ValueError, match="must precede"):
        run_test_filter(
            features,
            samples,
            validation,
            training,
        )


def test_training_filter_rejects_holdout():
    features, samples, training, _ = make_test_data()

    holdout = np.arange(
        TRAINING_COUNT + VALIDATION_COUNT,
        TOTAL_COUNT,
    )

    with pytest.raises(ValueError, match="Holdout observations"):
        run_test_filter(
            features,
            samples,
            training,
            holdout,
        )


def test_training_filter_rejects_duplicate_positions():
    features, samples, training, validation = make_test_data()

    duplicate_training = training.copy()
    duplicate_training[1] = duplicate_training[0]

    with pytest.raises(ValueError, match="Invalid chronological"):
        run_test_filter(
            features,
            samples,
            duplicate_training,
            validation,
        )
