import numpy as np
import pandas as pd

from earnings_surprise.evaluation.classification import run_ablations
from earnings_surprise.evaluation.ablation_features import FEATURE_SETS
from earnings_surprise.features import FEATURE_COLUMNS
from earnings_surprise.models.xgboost_model import build_xgboost_pipeline
from earnings_surprise.preprocessing import CONTINUOUS_COLUMNS

TEST_TREE_COUNT = 12


def make_test_data():
    """creates synthetic earnings observations across validation years"""
    year_counts = [
        (2019, 24),
        (2020, 8),
        (2021, 8),
        (2022, 8),
        (2023, 8),
        (2024, 8),
    ]

    dates = [
        pd.Timestamp(year=year, month=4, day=day)
        for year, count in year_counts
        for day in range(1, count + 1)
    ]

    sample_count = len(dates)
    positions = np.arange(sample_count)

    data = {
        column: np.sin(positions / (index + 2))
        for index, column in enumerate(CONTINUOUS_COLUMNS)
    }

    for lag in range(1, 5):
        data[f"beat_lag{lag}"] = (positions // (2 ** (lag - 1))) % 2

    historical_beats = np.column_stack([data[f"beat_lag{lag}"] for lag in range(1, 5)])

    data["beat_rate4"] = historical_beats.mean(axis=1)

    consecutive = np.ones(sample_count, dtype=bool)
    streaks = np.zeros(sample_count, dtype=int)

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

    samples = pd.DataFrame(
        {
            "ticker": ["TEST"] * sample_count,
            "target_date_pub": dates,
            "target_beat": positions % 2,
        }
    )

    return features, samples


def test_evaluate_feature_ablations(monkeypatch):
    features, samples = make_test_data()

    original_features = features.copy(deep=True)
    original_samples = samples.copy(deep=True)

    def build_small_model(feature_columns):
        model = build_xgboost_pipeline(feature_columns=feature_columns)

        model.set_params(
            classifier__n_estimators=TEST_TREE_COUNT,
            classifier__n_jobs=1,
        )

        return model

    monkeypatch.setattr(
        run_ablations,
        "build_xgboost_pipeline",
        build_small_model,
    )

    results = run_ablations.evaluate_feature_ablations(
        features,
        samples,
    )

    expected_column_counts = {
        "full_history": 39,
        "latest_event_only": 20,
        "without_trends": 36,
    }

    assert results["primary_metric"] == "macro_f1"
    assert results["holdout_used"] is False
    assert results["holdout_samples"] == 8
    assert len(results["folds"]) == 4

    for fold in results["folds"]:
        assert set(fold["models"]) == set(FEATURE_SETS)

        for name, metrics in fold["models"].items():
            assert metrics["samples"] == fold["validation_samples"]
            assert metrics["original_feature_count"] == len(FEATURE_SETS[name])
            assert metrics["transformed_feature_count"] == (
                expected_column_counts[name]
            )

            assert 0 <= metrics["macro_f1"] <= 1
            assert 0 <= metrics["accuracy"] <= 1
            assert 0 <= metrics["roc_auc"] <= 1

    assert set(results["mean_performance"]) == set(FEATURE_SETS)

    pd.testing.assert_frame_equal(features, original_features)
    pd.testing.assert_frame_equal(samples, original_samples)


def test_holdout_does_not_affect_ablation_results(monkeypatch):
    features, samples = make_test_data()

    def build_small_model(feature_columns):
        model = build_xgboost_pipeline(feature_columns=feature_columns)

        model.set_params(
            classifier__n_estimators=TEST_TREE_COUNT,
            classifier__n_jobs=1,
        )

        return model

    monkeypatch.setattr(
        run_ablations,
        "build_xgboost_pipeline",
        build_small_model,
    )

    original_results = run_ablations.evaluate_feature_ablations(
        features,
        samples,
    )

    modified_features = features.copy(deep=True)
    modified_samples = samples.copy(deep=True)

    holdout_mask = modified_samples["target_date_pub"] >= pd.Timestamp("2024-01-01")

    # changes only the reserved holdout observations
    modified_features.loc[holdout_mask, "actual_eps_lag1"] = 1_000_000.0

    modified_samples.loc[holdout_mask, "target_beat"] = (
        1 - modified_samples.loc[holdout_mask, "target_beat"]
    )

    modified_results = run_ablations.evaluate_feature_ablations(
        modified_features,
        modified_samples,
    )

    assert original_results == modified_results
