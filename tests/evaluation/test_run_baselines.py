import pandas as pd
import pytest

from earnings_surprise.evaluation.run_baselines import (
    BASELINE_MODELS,
    evaluate_baselines,
)


def make_test_data():
    samples = pd.DataFrame({
        "ticker": ["A"] * 6,
        "target_date_pub": pd.to_datetime([
            "2019-05-01",
            "2020-05-01",
            "2021-05-01",
            "2022-05-01",
            "2023-05-01",
            "2024-05-01",
        ]),
        "target_beat": [1, 0, 1, 0, 1, 0],
    })

    features = pd.DataFrame({
        "beat_lag1": [0, 1, 0, 1, 0, 1],
        "beat_rate4": [0.0, 0.5, 0.25, 1.0, 0.75, 0.5],
    })

    return features, samples


def test_evaluate_baselines():
    features, samples = make_test_data()

    original_features = features.copy(deep=True)
    original_samples = samples.copy(deep=True)

    results = evaluate_baselines(features, samples)

    assert results["experiment"] == (
        "chronological_baseline_evaluation"
    )

    assert results["primary_metric"] == "macro_f1"
    assert results["holdout_samples"] == 1
    assert len(results["folds"]) == 4

    for fold in results["folds"]:
        assert fold["validation_samples"] == 1
        assert set(fold["models"]) == set(BASELINE_MODELS)

        for metrics in fold["models"].values():
            assert metrics["samples"] == 1
            assert 0 <= metrics["accuracy"] <= 1
            assert 0 <= metrics["macro_f1"] <= 1

    pd.testing.assert_frame_equal(features, original_features)
    pd.testing.assert_frame_equal(samples, original_samples)


def test_holdout_outcomes_do_not_affect_baseline_results():
    features, samples = make_test_data()

    original_results = evaluate_baselines(features, samples)

    modified_samples = samples.copy()
    modified_samples.loc[5, "target_beat"] = 1

    modified_results = evaluate_baselines(
        features,
        modified_samples,
    )

    assert original_results == modified_results


def test_evaluate_baselines_requires_matching_rows():
    features, samples = make_test_data()

    features = features.iloc[:-1]

    with pytest.raises(ValueError, match="counts do not match"):
        evaluate_baselines(features, samples)


def test_evaluate_baselines_requires_historical_features():
    features, samples = make_test_data()

    features = features.drop(columns=["beat_rate4"])

    with pytest.raises(ValueError, match="Missing required features"):
        evaluate_baselines(features, samples)