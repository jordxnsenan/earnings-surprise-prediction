import numpy as np
import pandas as pd

from earnings_surprise.evaluation import final_holdout
from earnings_surprise.evaluation.ablation_features import FEATURE_SETS
from earnings_surprise.features import FEATURE_COLUMNS
from earnings_surprise.evaluation.final_holdout import (
    get_final_holdout_indices,
)


def make_test_samples():
    """creates unsorted observations spanning development and holdout"""

    return pd.DataFrame(
        {
            "ticker": ["A"] * 6,
            "target_date_pub": pd.to_datetime(
                [
                    "2024-03-01",
                    "2022-05-01",
                    "2019-10-01",
                    "2023-06-01",
                    "2020-04-01",
                    "2021-02-01",
                ]
            ),
            "target_beat": [1, 0, 1, 0, 1, 0],
        }
    )


def test_final_holdout_indices():
    samples = make_test_samples()

    training_indices, holdout_indices = get_final_holdout_indices(samples)

    training_years = samples.iloc[training_indices]["target_date_pub"].dt.year.tolist()

    holdout_years = samples.iloc[holdout_indices]["target_date_pub"].dt.year.tolist()

    assert sorted(training_years) == [2019, 2020, 2021, 2022, 2023]

    assert holdout_years == [2024]

    assert (
        np.intersect1d(
            training_indices,
            holdout_indices,
        ).size
        == 0
    )

    assert len(training_indices) + len(holdout_indices) == len(samples)


def test_evaluate_final_holdout(monkeypatch):
    """checks the selected model, training rows, and baseline results"""
    
    samples = make_test_samples()

    additional_holdout = pd.DataFrame(
        {
            "ticker": ["A"],
            "target_date_pub": [pd.Timestamp("2024-07-01")],
            "target_beat": [0],
        }
    )

    samples = pd.concat(
        [samples, additional_holdout],
        ignore_index=True,
    )

    features = pd.DataFrame(
        0.0,
        index=samples.index,
        columns=FEATURE_COLUMNS,
    )

    features["beat_lag1"] = [0, 0, 1, 0, 1, 0, 1]

    features["previous_report_month"] = pd.Categorical(
        [3, 5, 10, 6, 4, 2, 7],
        categories=range(1, 13),
    )

    features["previous_report_quarter"] = pd.Categorical(
        [1, 2, 4, 2, 2, 1, 3],
        categories=range(1, 5),
    )

    model_record = {}

    class RecordingModel:
        def fit(self, X, y):
            model_record["training_rows"] = X.index.tolist()
            model_record["training_labels"] = y.tolist()
            model_record["feature_columns"] = X.columns.tolist()

            return self

        def predict(self, X):
            model_record["holdout_rows"] = X.index.tolist()

            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return np.tile(
                [0.6, 0.4],
                (len(X), 1),
            )

    def build_recording_model(feature_columns):
        assert feature_columns == FEATURE_SETS["without_trends"]

        return RecordingModel()

    monkeypatch.setattr(
        final_holdout,
        "build_xgboost_pipeline",
        build_recording_model,
    )

    results = final_holdout.evaluate_final_holdout(
        features,
        samples,
    )

    assert results["primary_metric"] == "macro_f1"
    assert results["selected_feature_set"] == "without_trends"

    assert results["training_samples"] == 5
    assert results["holdout_samples"] == 2

    assert set(results["models"]) == {
        "xgboost_without_trends",
        "previous_outcome",
        "majority_class",
    }

    assert model_record["training_rows"] == [1, 2, 3, 4, 5]
    assert model_record["holdout_rows"] == [0, 6]
    assert model_record["training_labels"] == [0, 1, 0, 1, 0]

    assert model_record["feature_columns"] == (FEATURE_SETS["without_trends"])

    assert results["models"]["previous_outcome"]["confusion_matrix"] == [
        [0, 1],
        [1, 0],
    ]

    assert results["models"]["majority_class"]["confusion_matrix"] == [
        [1, 0],
        [1, 0],
    ]

    for metrics in results["models"].values():
        assert metrics["samples"] == 2
        assert 0 <= metrics["macro_f1"] <= 1
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1
