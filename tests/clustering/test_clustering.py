import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import make_blobs

from earnings_surprise.clustering.algorithms.kmeans import run_kmeans
from earnings_surprise.clustering.prepare import prepare_clustering_data
from earnings_surprise.features import FEATURE_COLUMNS
from earnings_surprise.preprocessing import CONTINUOUS_COLUMNS

TEST_SAMPLE_COUNT = 16
DEVELOPMENT_COUNT = 12
TEST_RANDOM_STATE = 42


def make_sample_data():
    """creates synthetic development and holdout observations"""
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
        pd.date_range("2024-01-01", periods=4)
    )

    samples = pd.DataFrame(
        {
            "ticker": ["A"] * TEST_SAMPLE_COUNT,
            "target_date_pub": dates,
            "target_beat": positions % 2,
        }
    )

    return features, samples


def test_clustering_preparation_excludes_holdout():
    features, samples = make_sample_data()

    original_features = features.copy(deep=True)

    transformed, positions, preprocessor = prepare_clustering_data(
        features,
        samples,
        sample_size=8,
    )

    assert transformed.shape == (8, 39)
    assert len(positions) == 8

    selected_dates = samples.iloc[positions]["target_date_pub"]

    assert selected_dates.lt("2024-01-01").all()
    assert np.isfinite(transformed).all()
    assert len(preprocessor.get_feature_names_out()) == 39

    pd.testing.assert_frame_equal(features, original_features)


def test_clustering_sampling_is_reproducible():
    features, samples = make_sample_data()

    first_data, first_positions, _ = prepare_clustering_data(
        features, samples, sample_size=8
    )

    second_data, second_positions, _ = prepare_clustering_data(
        features, samples, sample_size=8
    )

    np.testing.assert_array_equal(
        first_positions,
        second_positions,
    )

    np.testing.assert_allclose(first_data, second_data)


def test_holdout_changes_do_not_affect_clustering_data():
    features, samples = make_sample_data()

    original_data, original_positions, _ = prepare_clustering_data(
        features, samples, sample_size=8
    )

    modified_features = features.copy(deep=True)
    modified_samples = samples.copy(deep=True)

    modified_features.loc[DEVELOPMENT_COUNT:, "actual_eps_lag1"] = 1_000_000.0

    modified_samples.loc[DEVELOPMENT_COUNT:, "target_beat"] = 1

    modified_data, modified_positions, _ = prepare_clustering_data(
        modified_features,
        modified_samples,
        sample_size=8,
    )

    np.testing.assert_array_equal(
        original_positions,
        modified_positions,
    )

    np.testing.assert_allclose(original_data, modified_data)


def test_clustering_preparation_requires_aligned_rows():
    features, samples = make_sample_data()

    with pytest.raises(ValueError, match="counts do not match"):
        prepare_clustering_data(features.iloc[:-1], samples)


def test_kmeans_finds_separated_clusters():
    data, _ = make_blobs(
        n_samples=120,
        centers=4,
        n_features=5,
        cluster_std=0.3,
        random_state=TEST_RANDOM_STATE,
    )

    labels, summary = run_kmeans(data, cluster_count=4)

    assert len(labels) == 120
    assert summary["cluster_count"] == 4
    assert sum(summary["cluster_sizes"].values()) == 120
    assert summary["silhouette_score"] > 0.7
    assert summary["calinski_harabasz_score"] > 0
    assert summary["davies_bouldin_score"] >= 0


def test_kmeans_rejects_invalid_cluster_count():
    data = np.ones((5, 3))

    with pytest.raises(ValueError, match="Invalid number"):
        run_kmeans(data, cluster_count=6)
