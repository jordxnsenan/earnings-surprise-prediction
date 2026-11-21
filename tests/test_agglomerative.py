import numpy as np
import pytest

from sklearn.datasets import make_blobs

from earnings_surprise.clustering.agglomerative import (
    INITIAL_CLUSTER_COUNT,
    run_agglomerative,
)

TEST_SAMPLE_COUNT = 120
TEST_RANDOM_STATE = 42


def make_test_data():
    """creates two well-separated synthetic clusters"""
    data, _ = make_blobs(
        n_samples=TEST_SAMPLE_COUNT,
        centers=2,
        n_features=5,
        cluster_std=0.35,
        random_state=TEST_RANDOM_STATE,
    )

    return data


def test_agglomerative_finds_separated_clusters():
    data = make_test_data()
    original_data = data.copy()

    labels, summary = run_agglomerative(data)

    assert len(labels) == TEST_SAMPLE_COUNT
    assert summary["algorithm"] == "agglomerative"
    assert summary["linkage"] == "ward"
    assert summary["cluster_count"] == INITIAL_CLUSTER_COUNT

    assert sum(summary["cluster_sizes"].values()) == (TEST_SAMPLE_COUNT)

    assert summary["silhouette_score"] > 0.7
    assert summary["calinski_harabasz_score"] > 0
    assert summary["davies_bouldin_score"] >= 0

    np.testing.assert_array_equal(data, original_data)


def test_agglomerative_is_reproducible():
    data = make_test_data()

    first_labels, first_summary = run_agglomerative(data)
    second_labels, second_summary = run_agglomerative(data)

    np.testing.assert_array_equal(
        first_labels,
        second_labels,
    )

    assert first_summary == second_summary


def test_agglomerative_supports_multiple_cluster_counts():
    data = make_test_data()

    for cluster_count in (2, 3, 4):
        labels, summary = run_agglomerative(
            data,
            cluster_count=cluster_count,
        )

        assert len(np.unique(labels)) == cluster_count
        assert summary["cluster_count"] == cluster_count
        assert sum(summary["cluster_sizes"].values()) == len(data)


@pytest.mark.parametrize(
    "invalid_count",
    [0, 1, TEST_SAMPLE_COUNT, -1, True, 2.5],
)
def test_agglomerative_rejects_invalid_cluster_count(invalid_count):
    data = make_test_data()

    with pytest.raises(ValueError, match="Invalid number"):
        run_agglomerative(data, cluster_count=invalid_count)


def test_agglomerative_rejects_nonfinite_data():
    data = make_test_data()
    data[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite values"):
        run_agglomerative(data)


def test_agglomerative_requires_two_dimensional_data():
    data = np.arange(20, dtype=float)

    with pytest.raises(ValueError, match="2D matrix"):
        run_agglomerative(data)
