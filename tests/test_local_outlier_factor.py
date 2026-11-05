import numpy as np
import pytest

from earnings_surprise.outliers.local_outlier_factor import (
    OUTLIER_LABEL,
    run_local_outlier_factor,
)

TEST_RANDOM_STATE = 42
TEST_NEIGHBOR_COUNT = 20
TEST_CONTAMINATION = 0.05


def make_test_data():
    """creates ordinary observations and isolated extreme points"""
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


def test_lof_detects_extreme_observations():
    data = make_test_data()
    original_data = data.copy()

    labels, scores, summary = run_local_outlier_factor(
        data,
        neighbor_count=TEST_NEIGHBOR_COUNT,
        contamination=TEST_CONTAMINATION,
    )

    assert len(labels) == len(data)
    assert len(scores) == len(data)

    assert summary["algorithm"] == "local_outlier_factor"
    assert summary["samples"] == len(data)
    assert summary["feature_count"] == data.shape[1]
    assert summary["neighbor_count"] == TEST_NEIGHBOR_COUNT

    assert summary["outlier_count"] > 0
    assert summary["inlier_count"] > 0

    assert summary["outlier_count"] + summary["inlier_count"] == len(data)

    # extreme observations should be flagged
    assert np.all(labels[-4:] == OUTLIER_LABEL)

    # lower scores correspond to outlier assignments
    np.testing.assert_array_equal(
        labels == OUTLIER_LABEL,
        scores < summary["score_cutoff"],
    )

    np.testing.assert_array_equal(data, original_data)


def test_lof_is_reproducible():
    data = make_test_data()

    first_labels, first_scores, first_summary = run_local_outlier_factor(data)

    second_labels, second_scores, second_summary = run_local_outlier_factor(data)

    np.testing.assert_array_equal(
        first_labels,
        second_labels,
    )

    np.testing.assert_allclose(
        first_scores,
        second_scores,
    )

    assert first_summary == second_summary


def test_lof_scores_are_finite():
    data = make_test_data()

    _, scores, summary = run_local_outlier_factor(data)

    assert np.isfinite(scores).all()
    assert all(np.isfinite(score) for score in summary["score_percentiles"].values())


@pytest.mark.parametrize(
    "invalid_neighbor_count",
    [0, 1, 204, True, 2.5],
)
def test_lof_rejects_invalid_neighbor_count(
    invalid_neighbor_count,
):
    data = make_test_data()

    with pytest.raises(ValueError, match="Invalid neighbor count"):
        run_local_outlier_factor(
            data,
            neighbor_count=invalid_neighbor_count,
        )


@pytest.mark.parametrize(
    "invalid_contamination",
    [0.0, -0.1, 0.6, True, float("nan")],
)
def test_lof_rejects_invalid_contamination(
    invalid_contamination,
):
    data = make_test_data()

    with pytest.raises(ValueError, match="Invalid contamination"):
        run_local_outlier_factor(
            data,
            contamination=invalid_contamination,
        )


def test_lof_rejects_nonfinite_data():
    data = make_test_data()
    data[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite values"):
        run_local_outlier_factor(data)


def test_lof_requires_two_dimensional_data():
    data = np.arange(20, dtype=float)

    with pytest.raises(ValueError, match="valid 2D matrix"):
        run_local_outlier_factor(data)
