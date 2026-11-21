import numpy as np
import pytest

from sklearn.datasets import make_blobs

from earnings_surprise.clustering.algorithms.dbscan import run_dbscan
from earnings_surprise.clustering.experiments.dbscan_sweep import (
    build_parameter_grid,
    evaluate_dbscan_grid,
)

TEST_RANDOM_STATE = 42


def make_test_data():
    """Create synthetic clusters with two isolated noise observations."""
    data, _ = make_blobs(
        n_samples=120,
        centers=[[-5, -5], [0, 5], [5, -5]],
        cluster_std=0.25,
        random_state=TEST_RANDOM_STATE,
    )

    noise = np.array(
        [
            [100.0, 100.0],
            [-100.0, -100.0],
        ]
    )

    return np.vstack([data, noise])


def test_dbscan_detects_clusters_and_noise():
    data = make_test_data()
    original_data = data.copy()

    labels, summary = run_dbscan(
        data,
        eps=0.8,
        min_samples=5,
    )

    assert len(labels) == len(data)
    assert summary["cluster_count"] == 3
    assert summary["noise_samples"] == 2
    assert summary["clustered_samples"] == 120
    assert summary["silhouette_score"] > 0.7
    assert summary["calinski_harabasz_score"] > 0
    assert summary["davies_bouldin_score"] >= 0

    assert sum(summary["cluster_sizes"].values()) == 120

    np.testing.assert_array_equal(data, original_data)


def test_dbscan_handles_all_noise():
    data = np.arange(30, dtype=float).reshape(10, 3) * 20

    labels, summary = run_dbscan(
        data,
        eps=0.1,
        min_samples=3,
    )

    assert np.all(labels == -1)
    assert summary["cluster_count"] == 0
    assert summary["noise_samples"] == 10
    assert summary["silhouette_score"] is None
    assert summary["calinski_harabasz_score"] is None
    assert summary["davies_bouldin_score"] is None


def test_dbscan_handles_single_cluster():
    data = np.zeros((20, 3))

    _, summary = run_dbscan(
        data,
        eps=0.5,
        min_samples=4,
    )

    assert summary["cluster_count"] == 1
    assert summary["noise_samples"] == 0
    assert summary["silhouette_score"] is None


def test_dbscan_is_reproducible():
    data = make_test_data()

    first_labels, first_summary = run_dbscan(data, 0.8, 5)
    second_labels, second_summary = run_dbscan(data, 0.8, 5)

    np.testing.assert_array_equal(first_labels, second_labels)
    assert first_summary == second_summary


@pytest.mark.parametrize("invalid_eps", [0, -1, float("nan")])
def test_dbscan_rejects_invalid_eps(invalid_eps):
    data = make_test_data()

    with pytest.raises(ValueError, match="eps must be"):
        run_dbscan(data, invalid_eps, 5)


@pytest.mark.parametrize("invalid_count", [0, 1, 200, True])
def test_dbscan_rejects_invalid_min_samples(invalid_count):
    data = make_test_data()

    with pytest.raises(ValueError, match="Invalid min_samples"):
        run_dbscan(data, 0.8, invalid_count)


def test_dbscan_parameter_grid():
    diagnostics = {
        "holdout_used": False,
        "diagnostics": [
            {
                "min_samples": 10,
                "distance_percentiles": {
                    "50": 2.2,
                    "70": 2.6,
                    "85": 3.2,
                    "95": 4.3,
                },
            }
        ],
    }

    grid = build_parameter_grid(diagnostics)

    assert len(grid) == 4
    assert grid[0]["min_samples"] == 10
    assert grid[0]["eps"] == 2.2
    assert grid[-1]["eps_percentile"] == 95


def test_dbscan_grid_rejects_holdout_usage():
    diagnostics = {
        "holdout_used": True,
        "diagnostics": [],
    }

    with pytest.raises(ValueError, match="exclude the holdout"):
        build_parameter_grid(diagnostics)


def test_dbscan_grid_evaluates_multiple_parameters():
    data = make_test_data()

    configurations = [
        {"eps": 0.7, "min_samples": 5, "eps_percentile": 50},
        {"eps": 0.8, "min_samples": 5, "eps_percentile": 70},
    ]

    results = evaluate_dbscan_grid(data, configurations)

    assert len(results) == 2
    assert all(result["samples"] == len(data) for result in results)
    assert results[0]["eps_percentile"] == 50
    assert results[1]["eps_percentile"] == 70
