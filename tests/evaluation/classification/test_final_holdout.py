import json
import numpy as np
import pandas as pd

from earnings_surprise.evaluation.classification import final_holdout
from earnings_surprise.evaluation.ablation_features import FEATURE_SETS
from earnings_surprise.features import FEATURE_COLUMNS
from earnings_surprise.evaluation.classification.final_holdout import (
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


def test_holdout_changes_do_not_affect_training(monkeypatch):
    samples = make_test_samples()

    features = pd.DataFrame(
        0.0,
        index=samples.index,
        columns=FEATURE_COLUMNS,
    )

    features["beat_lag1"] = [0, 0, 1, 0, 1, 0]

    recorded_training_data = []

    class RecordingModel:
        def fit(self, X, y):
            recorded_training_data.append(
                (
                    X.copy(deep=True),
                    y.copy(deep=True),
                )
            )
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return np.tile([0.6, 0.4], (len(X), 1))

    def build_recording_model(feature_columns):
        assert feature_columns == FEATURE_SETS["without_trends"]
        return RecordingModel()

    monkeypatch.setattr(
        final_holdout,
        "build_xgboost_pipeline",
        build_recording_model,
    )

    final_holdout.evaluate_final_holdout(features, samples)

    modified_features = features.copy(deep=True)
    modified_samples = samples.copy(deep=True)

    holdout_mask = modified_samples["target_date_pub"] >= pd.Timestamp("2024-01-01")

    modified_features.loc[holdout_mask, "actual_eps_lag1"] = 1_000_000.0

    modified_features.loc[holdout_mask, "beat_lag1"] = 1

    modified_samples.loc[holdout_mask, "target_beat"] = (
        1 - modified_samples.loc[holdout_mask, "target_beat"]
    )

    final_holdout.evaluate_final_holdout(
        modified_features,
        modified_samples,
    )

    assert len(recorded_training_data) == 2

    first_features, first_targets = recorded_training_data[0]
    second_features, second_targets = recorded_training_data[1]

    pd.testing.assert_frame_equal(
        first_features,
        second_features,
    )

    pd.testing.assert_series_equal(
        first_targets,
        second_targets,
    )


def test_main_saves_final_results(tmp_path, monkeypatch, capsys):
    """checks that synthetic holdout results are saved correctly"""

    samples = make_test_samples()

    features = pd.DataFrame(
        0.0,
        index=samples.index,
        columns=FEATURE_COLUMNS,
    )

    features["beat_lag1"] = [0, 0, 1, 0, 1, 0]

    feature_path = tmp_path / "features.parquet"
    sample_path = tmp_path / "samples.parquet"
    results_path = tmp_path / "final_holdout_results.json"
    predictions_path = tmp_path / "final_holdout_predictions.csv"

    features.to_parquet(feature_path)
    samples.to_parquet(sample_path)

    monkeypatch.setattr(final_holdout, "FEATURE_MATRIX_PATH", feature_path)
    monkeypatch.setattr(final_holdout, "PREDICTION_WINDOWS_PATH", sample_path)
    monkeypatch.setattr(final_holdout, "FINAL_RESULTS_PATH", results_path)
    monkeypatch.setattr(final_holdout, "FINAL_PREDICTIONS_PATH", predictions_path)

    training_rows = []

    class RecordingModel:
        def fit(self, X, y):
            training_rows.extend(X.index.tolist())
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return np.tile([0.6, 0.4], (len(X), 1))

    def build_recording_model(feature_columns):
        assert feature_columns == FEATURE_SETS["without_trends"]
        return RecordingModel()

    monkeypatch.setattr(
        final_holdout,
        "build_xgboost_pipeline",
        build_recording_model,
    )

    final_holdout.main()

    assert results_path.is_file()

    results = json.loads(results_path.read_text(encoding="utf-8"))

    assert results["experiment"] == "final_2024_holdout"
    assert results["selected_feature_set"] == "without_trends"
    assert results["training_samples"] == 5
    assert results["holdout_samples"] == 1

    assert training_rows == [1, 2, 3, 4, 5]

    assert set(results["models"]) == {
        "xgboost_without_trends",
        "previous_outcome",
        "majority_class",
    }

    for metrics in results["models"].values():
        assert metrics["samples"] == 1
        assert metrics["roc_auc"] is None

    output = capsys.readouterr().out

    assert "Final 2024 Holdout Evaluation" in output
    assert "ROC-AUC=N/A" in output

    assert predictions_path.is_file()

    predictions = pd.read_csv(predictions_path)

    assert len(predictions) == 1

    assert predictions.loc[0, "ticker"] == "A"
    assert predictions.loc[0, "target_date_pub"] == "2024-03-01"
    assert predictions.loc[0, "target_beat"] == 1

    assert predictions.loc[0, "xgboost_without_trends_prediction"] == 0

    assert np.isclose(
        predictions.loc[0, "xgboost_without_trends_beat_probability"],
        0.4,
    )

    assert "previous_outcome_prediction" in predictions.columns
    assert "majority_class_prediction" in predictions.columns

    assert "holdout_predictions" not in results
