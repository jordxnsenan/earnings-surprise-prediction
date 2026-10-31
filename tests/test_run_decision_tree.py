import numpy as np
import pandas as pd
import pytest

from earnings_surprise.evaluation.run_decision_tree import (
    MODEL_CONFIGURATIONS,
    evaluate_decision_tree_models,
)
from earnings_surprise.features import FEATURE_COLUMNS
from earnings_surprise.preprocessing import (
    CONTINUOUS_COLUMNS,
)


def make_test_data():
    """creates synthetic earnings features spanning validation years"""
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

    row_count = len(dates)
    positions = np.arange(row_count)

    data = {
        column: np.sin(positions / (index + 2))
        for index, column in enumerate(CONTINUOUS_COLUMNS)
    }

    for lag in range(1, 5):
        data[f"beat_lag{lag}"] = (
            positions // (2 ** (lag - 1))
        ) % 2

    historical_beats = np.column_stack([
        data[f"beat_lag{lag}"]
        for lag in range(1, 5)
    ])

    data["beat_rate4"] = historical_beats.mean(axis=1)

    streaks = np.zeros(row_count, dtype=int)
    consecutive = np.ones(row_count, dtype=bool)

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

    samples = pd.DataFrame({
        "ticker": ["A"] * row_count,
        "target_date_pub": dates,
        "target_beat": positions % 2,
    })

    return features, samples


def test_evaluate_decision_tree_models():
    features, samples = make_test_data()

    original_features = features.copy(deep=True)
    original_samples = samples.copy(deep=True)

    results = evaluate_decision_tree_models(features, samples)

    assert results["primary_metric"] == "macro_f1"
    assert results["holdout_samples"] == 8
    assert len(results["folds"]) == 4

    for fold in results["folds"]:
        assert set(fold["models"]) == set(MODEL_CONFIGURATIONS)

        all_features = fold["models"]["all_features"]
        selected = fold["models"]["mi_selection"]

        assert all_features["feature_count"] == 39
        assert selected["feature_count"] == 15

        assert len(all_features["selected_features"]) == 0
        assert len(selected["selected_features"]) == 15

        assert len(all_features["feature_importances"]) == 39
        assert len(selected["feature_importances"]) == 15

        for model_results in fold["models"].values():
            assert model_results["samples"] == fold["validation_samples"]
            assert model_results["tree_depth"] <= 6
            assert model_results["leaf_count"] >= 1
            assert 0 <= model_results["macro_f1"] <= 1

    assert set(results["mean_performance"]) == set(MODEL_CONFIGURATIONS)

    pd.testing.assert_frame_equal(features, original_features)
    pd.testing.assert_frame_equal(samples, original_samples)


def test_holdout_does_not_affect_tree_results():
    features, samples = make_test_data()

    original_results = evaluate_decision_tree_models(
        features, samples
    )

    modified_features = features.copy(deep=True)
    modified_samples = samples.copy(deep=True)

    holdout_mask = (
        modified_samples["target_date_pub"]
        >= pd.Timestamp("2024-01-01")
    )

    modified_features.loc[
        holdout_mask, "actual_eps_lag1"
    ] = 1_000_000.0

    modified_samples.loc[
        holdout_mask, "target_beat"
    ] = 1 - modified_samples.loc[holdout_mask, "target_beat"]

    modified_results = evaluate_decision_tree_models(
        modified_features,
        modified_samples,
    )

    assert modified_results == original_results


def test_decision_tree_runner_requires_matching_rows():
    features, samples = make_test_data()

    with pytest.raises(ValueError, match="counts do not match"):
        evaluate_decision_tree_models(
            features.iloc[:-1],
            samples,
        )


def test_decision_tree_runner_requires_complete_features():
    features, samples = make_test_data()

    features = features.drop(columns=["beat_rate4"])

    with pytest.raises(ValueError, match="do not match"):
        evaluate_decision_tree_models(features, samples)