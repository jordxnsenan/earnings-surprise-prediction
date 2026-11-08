import numpy as np
import pandas as pd
import pytest

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from earnings_surprise.features import FEATURE_COLUMNS
from earnings_surprise.tuning import xgboost_search as tuning


def make_test_data():
    """creates development, assessment and holdout observations"""
    dates = []

    for year, count in (
        (2019, 8),
        (2020, 4),
        (2021, 4),
        (2022, 4),
        (2023, 4),
        (2024, 4),
    ):
        dates.extend(
            pd.date_range(
                f"{year}-01-02",
                periods=count,
            )
        )

    row_count = len(dates)

    features = pd.DataFrame(
        {column: np.arange(row_count, dtype=float) for column in FEATURE_COLUMNS}
    )

    samples = pd.DataFrame(
        {
            "ticker": ["TEST"] * row_count,
            "target_date_pub": dates,
            "target_beat": np.arange(row_count) % 2,
        }
    )

    return features, samples


def test_search_space_uses_existing_pipeline_step():
    pipeline = Pipeline(
        [
            (
                "classifier",
                XGBClassifier(
                    n_estimators=100,
                    random_state=42,
                ),
            )
        ]
    )

    search_space, estimator_name = tuning.build_search_space(pipeline)

    assert estimator_name == "classifier"
    assert "classifier__n_estimators" in search_space
    assert "classifier__max_depth" in search_space

    for name in search_space:
        assert name in pipeline.get_params()


def test_search_excludes_assessment_and_holdout(monkeypatch):
    features, samples = make_test_data()

    original_features = features.copy(deep=True)
    original_samples = samples.copy(deep=True)

    # uses lightweight classifier to keep integration test quick
    monkeypatch.setattr(
        tuning,
        "build_xgboost_pipeline",
        lambda selected_feature_count: Pipeline([("classifier", DummyClassifier())]),
    )

    monkeypatch.setattr(
        tuning,
        "build_search_space",
        lambda pipeline: (
            {
                "classifier__strategy": [
                    "prior",
                    "most_frequent",
                ]
            },
            "classifier",
        ),
    )

    report = tuning.run_xgboost_search(
        features,
        samples,
        iterations=2,
    )

    assert report["search_samples"] == 20
    assert report["assessment_samples"] == 4
    assert report["holdout_samples"] == 4

    assert report["search_validation_years"] == [2020, 2021, 2022]

    assert report["assessment_used"] is False
    assert report["holdout_used"] is False
    assert report["refit_during_search"] is False

    assert len(report["candidates"]) == 2
    assert report["best_configuration"]["mean_macro_f1"] >= 0

    pd.testing.assert_frame_equal(
        features,
        original_features,
    )

    pd.testing.assert_frame_equal(
        samples,
        original_samples,
    )


def test_search_works_with_shuffled_observations(monkeypatch):
    features, samples = make_test_data()

    permutation = np.random.default_rng(42).permutation(len(samples))

    features = features.iloc[permutation].reset_index(drop=True)
    samples = samples.iloc[permutation].reset_index(drop=True)

    monkeypatch.setattr(
        tuning,
        "build_xgboost_pipeline",
        lambda selected_feature_count: Pipeline([("classifier", DummyClassifier())]),
    )

    monkeypatch.setattr(
        tuning,
        "build_search_space",
        lambda pipeline: (
            {"classifier__strategy": ["prior"]},
            "classifier",
        ),
    )

    report = tuning.run_xgboost_search(
        features,
        samples,
        iterations=1,
    )

    assert report["search_samples"] == 20
    assert report["holdout_used"] is False
    assert len(report["candidates"]) == 1


@pytest.mark.parametrize(
    "invalid_iterations",
    [0, -1, True, 2.5],
)
def test_search_rejects_invalid_iterations(
    invalid_iterations,
):
    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        tuning.run_xgboost_search(
            pd.DataFrame(),
            pd.DataFrame(),
            iterations=invalid_iterations,
        )


def test_search_rejects_misaligned_features():
    features, samples = make_test_data()

    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        tuning.run_xgboost_search(
            features.iloc[:-1],
            samples,
        )
