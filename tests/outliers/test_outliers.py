import numpy as np
import pandas as pd
import pytest

from earnings_surprise.features import FEATURE_COLUMNS
from earnings_surprise.outliers.algorithms.isolation_forest import (
    ISOLATION_OUTLIER_LABEL,
    run_isolation_forest,
)
from earnings_surprise.outliers.prepare import prepare_outlier_data
from earnings_surprise.preprocessing import CONTINUOUS_COLUMNS

DEVELOPMENT_COUNT = 16
HOLDOUT_COUNT = 4
TEST_SAMPLE_COUNT = DEVELOPMENT_COUNT + HOLDOUT_COUNT
TEST_TREE_COUNT = 32
TEST_RANDOM_STATE = 42


def make_sample_data():
    """creates synthetic development and holdout earnings histories"""
    positions = np.arange(TEST_SAMPLE_COUNT)

    data = {
        column: np.sin(positions / (index + 2))
        for index, column in enumerate(CONTINUOUS_COLUMNS)
    }

    for lag in range(1, 5):
        data[f"beat_lag{lag}"] = (positions // (2 ** (lag - 1))) % 2

    historical_beats = np.column_stack([data[f"beat_lag{lag}"] for lag in range(1, 5)])

    data["beat_rate4"] = historical_beats.mean(axis=1)

    streaks = np.zeros(TEST_SAMPLE_COUNT, dtype=int)
    consecutive = np.ones(TEST_SAMPLE_COUNT, dtype=bool)

    for lag in range(4):
        consecutive &= historical_beats[:, lag] == 1
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

    dates = list(pd.date_range("2022-01-01", periods=DEVELOPMENT_COUNT)) + list(
        pd.date_range("2024-01-01", periods=HOLDOUT_COUNT)
    )

    samples = pd.DataFrame(
        {
            "target_date_pub": dates,
            "target_beat": positions % 2,
        }
    )

    return features, samples


def make_anomaly_data():
    """creates ordinary observations with several extreme points"""
    generator = np.random.default_rng(TEST_RANDOM_STATE)

    ordinary = generator.normal(
        loc=0.0,
        scale=0.4,
        size=(200, 4),
    )

    unusual = np.array(
        [
            [9.0, 9.0, 9.0, 9.0],
            [-9.0, -9.0, -9.0, -9.0],
            [9.0, -9.0, 9.0, -9.0],
            [-9.0, 9.0, -9.0, 9.0],
        ]
    )

    return np.vstack([ordinary, unusual])


def test_outlier_preparation_excludes_holdout():
    features, samples = make_sample_data()
    original_features = features.copy(deep=True)

    transformed, positions, preprocessor = prepare_outlier_data(features, samples)

    assert transformed.shape == (DEVELOPMENT_COUNT, 39)
    assert positions.tolist() == list(range(DEVELOPMENT_COUNT))
    assert len(preprocessor.get_feature_names_out()) == 39
    assert np.isfinite(transformed).all()

    pd.testing.assert_frame_equal(features, original_features)


def test_outlier_sampling_is_reproducible():
    features, samples = make_sample_data()

    first_data, first_positions, _ = prepare_outlier_data(
        features, samples, sample_size=8
    )

    second_data, second_positions, _ = prepare_outlier_data(
        features, samples, sample_size=8
    )

    np.testing.assert_array_equal(first_positions, second_positions)
    np.testing.assert_allclose(first_data, second_data)


def test_holdout_changes_do_not_affect_outlier_preparation():
    features, samples = make_sample_data()

    original_data, original_positions, _ = prepare_outlier_data(features, samples)

    changed_features = features.copy(deep=True)
    changed_samples = samples.copy(deep=True)

    changed_features.loc[DEVELOPMENT_COUNT:, "actual_eps_lag1"] = 1_000_000.0

    changed_samples.loc[DEVELOPMENT_COUNT:, "target_beat"] = 1

    changed_data, changed_positions, _ = prepare_outlier_data(
        changed_features, changed_samples
    )

    np.testing.assert_array_equal(original_positions, changed_positions)

    np.testing.assert_allclose(original_data, changed_data)


def test_outlier_preparation_rejects_misaligned_rows():
    features, samples = make_sample_data()

    with pytest.raises(ValueError, match="counts do not match"):
        prepare_outlier_data(features.iloc[:-1], samples)


def test_isolation_forest_detects_unusual_observations():
    data = make_anomaly_data()
    original_data = data.copy()

    labels, scores, summary = run_isolation_forest(
        data,
        contamination=0.05,
        tree_count=TEST_TREE_COUNT,
    )

    assert len(labels) == len(data)
    assert len(scores) == len(data)

    assert summary["tree_count"] == TEST_TREE_COUNT
    assert summary["samples"] == len(data)

    assert summary["outlier_count"] > 0
    assert summary["inlier_count"] > 0

    assert summary["outlier_count"] + summary["inlier_count"] == len(data)

    # flagged obs. should have lower anomaly scores
    assert scores[labels == ISOLATION_OUTLIER_LABEL].mean() < (
        scores[labels == 1].mean()
    )

    np.testing.assert_array_equal(data, original_data)


def test_isolation_forest_is_reproducible():
    data = make_anomaly_data()

    first_labels, first_scores, first_summary = run_isolation_forest(
        data, tree_count=TEST_TREE_COUNT
    )

    second_labels, second_scores, second_summary = run_isolation_forest(
        data, tree_count=TEST_TREE_COUNT
    )

    np.testing.assert_array_equal(first_labels, second_labels)
    np.testing.assert_allclose(first_scores, second_scores)

    assert first_summary == second_summary


@pytest.mark.parametrize(
    "invalid_contamination",
    [0.0, -0.1, 0.6, True, float("nan")],
)
def test_isolation_forest_rejects_invalid_contamination(
    invalid_contamination,
):
    data = make_anomaly_data()

    with pytest.raises(ValueError, match="Invalid contamination"):
        run_isolation_forest(
            data,
            contamination=invalid_contamination,
            tree_count=TEST_TREE_COUNT,
        )


def test_isolation_forest_rejects_nonfinite_data():
    data = make_anomaly_data()
    data[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite values"):
        run_isolation_forest(data, tree_count=TEST_TREE_COUNT)
