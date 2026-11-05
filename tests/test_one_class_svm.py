import numpy as np
import pytest

from earnings_surprise.outliers.one_class_svm import (
    OUTLIER_LABEL,
    run_one_class_svm,
)

TEST_RANDOM_STATE = 42
TEST_NU = 0.05


def make_test_data():
    """generates ordinary observations and extreme anomalies"""
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


def test_one_class_svm_detects_extreme_observations():
    data = make_test_data()
    original_data = data.copy()

    labels, scores, summary = run_one_class_svm(
        data,
        nu=TEST_NU,
    )

    assert len(labels) == len(data)
    assert len(scores) == len(data)

    assert summary["algorithm"] == "one_class_svm"
    assert summary["samples"] == len(data)
    assert summary["feature_count"] == data.shape[1]
    assert summary["kernel"] == "rbf"
    assert summary["nu"] == TEST_NU

    assert summary["outlier_count"] > 0
    assert summary["inlier_count"] > 0
    assert summary["support_vector_count"] > 0

    assert summary["outlier_count"] + summary["inlier_count"] == len(data)

    # extreme observations should be anomalous
    assert np.all(labels[-4:] == OUTLIER_LABEL)

    np.testing.assert_array_equal(
        labels == OUTLIER_LABEL,
        scores < summary["score_threshold"],
    )

    np.testing.assert_array_equal(data, original_data)


def test_one_class_svm_is_reproducible():
    data = make_test_data()

    first_labels, first_scores, first_summary = run_one_class_svm(data)

    second_labels, second_scores, second_summary = run_one_class_svm(data)

    np.testing.assert_array_equal(
        first_labels,
        second_labels,
    )

    np.testing.assert_allclose(
        first_scores,
        second_scores,
    )

    assert first_summary == second_summary


def test_one_class_svm_scores_are_finite():
    data = make_test_data()

    _, scores, summary = run_one_class_svm(data)

    assert np.isfinite(scores).all()

    assert all(np.isfinite(score) for score in summary["score_percentiles"].values())


@pytest.mark.parametrize(
    "invalid_nu",
    [0.0, -0.1, 1.1, True, float("nan"), "invalid"],
)
def test_one_class_svm_rejects_invalid_nu(invalid_nu):
    data = make_test_data()

    with pytest.raises(ValueError, match="Invalid nu"):
        run_one_class_svm(data, nu=invalid_nu)


def test_one_class_svm_rejects_nonfinite_data():
    data = make_test_data()
    data[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite values"):
        run_one_class_svm(data)


def test_one_class_svm_requires_two_dimensional_data():
    data = np.arange(20, dtype=float)

    with pytest.raises(ValueError, match="valid 2D matrix"):
        run_one_class_svm(data)
