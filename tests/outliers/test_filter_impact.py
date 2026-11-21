import numpy as np
import pandas as pd
import pytest

from earnings_surprise.features import FEATURE_COLUMNS
from earnings_surprise.outliers import evaluate_filter_impact as impact

TEST_ROW_COUNT = 24
TRAINING_COUNT = 12
VALIDATION_COUNT = 8
HOLDOUT_COUNT = 4


def make_test_data():
    """creates a small dataset with chronological validation rows"""
    features = pd.DataFrame(
        {column: np.arange(TEST_ROW_COUNT, dtype=float) for column in FEATURE_COLUMNS}
    )

    dates = (
        list(pd.date_range("2019-01-01", periods=TRAINING_COUNT))
        + list(pd.date_range("2020-01-01", periods=VALIDATION_COUNT))
        + list(pd.date_range("2024-01-01", periods=HOLDOUT_COUNT))
    )

    samples = pd.DataFrame(
        {
            "target_date_pub": dates,
            "target_beat": np.arange(TEST_ROW_COUNT) % 2,
        }
    )

    return features, samples


def test_classification_metrics():
    actual = np.array([0, 0, 1, 1])
    predicted = np.array([0, 1, 1, 1])
    probabilities = np.array([0.1, 0.65, 0.8, 0.9])

    metrics = impact.calculate_classification_metrics(
        actual,
        predicted,
        probabilities,
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx(0.7333333)
    assert metrics["roc_auc"] == pytest.approx(1.0)

    assert metrics["recall_class_0"] == pytest.approx(0.5)
    assert metrics["recall_class_1"] == pytest.approx(1.0)

    assert metrics["confusion_matrix"] == [
        [1, 1],
        [0, 2],
    ]


def test_classification_metrics_reject_mismatched_lengths():
    actual = np.array([0, 1, 0])
    predicted = np.array([0, 1])
    probabilities = np.array([0.2, 0.8, 0.3])

    with pytest.raises(ValueError, match="matching lengths"):
        impact.calculate_classification_metrics(
            actual,
            predicted,
            probabilities,
        )


def test_classification_metrics_reject_invalid_probabilities():
    actual = np.array([0, 1])
    predicted = np.array([0, 1])
    probabilities = np.array([-0.1, 1.1])

    with pytest.raises(ValueError, match="Invalid classification"):
        impact.calculate_classification_metrics(
            actual,
            predicted,
            probabilities,
        )


def test_filter_impact_preserves_validation_rows(monkeypatch):
    """verifies independent training fits and identical validation rows"""
    features, samples = make_test_data()

    training_positions = np.arange(TRAINING_COUNT)
    validation_positions = np.arange(
        TRAINING_COUNT,
        TRAINING_COUNT + VALIDATION_COUNT,
    )
    holdout_positions = np.arange(
        TRAINING_COUNT + VALIDATION_COUNT,
        TEST_ROW_COUNT,
    )

    folds = [
        {
            "validation_year": 2020,
            "training_indices": training_positions,
            "validation_indices": validation_positions,
        }
    ]

    monkeypatch.setattr(
        impact,
        "create_chronological_splits",
        lambda data: (folds, holdout_positions),
    )

    def fake_training_filter(
        input_features,
        input_samples,
        training_indices,
        validation_indices,
    ):
        np.testing.assert_array_equal(
            training_indices,
            training_positions,
        )

        np.testing.assert_array_equal(
            validation_indices,
            validation_positions,
        )

        retained = training_indices[:-2]

        return retained, {
            "removed_training_samples": 2,
            "removed_fraction": 2 / len(training_indices),
            "validation_filtered": False,
            "holdout_used": False,
        }

    monkeypatch.setattr(
        impact,
        "filter_training_fold",
        fake_training_filter,
    )

    fitted_models = []

    class RecordingClassifier:
        """small classifier double that records training and validation rows"""

        def __init__(self):
            self.classes_ = np.array([0, 1])
            self.record = {}

        def fit(self, X, y):
            self.record["training_rows"] = X.index.tolist()
            self.record["training_labels"] = list(y)
            fitted_models.append(self.record)
            return self

        def predict(self, X):
            self.record["validation_rows"] = X.index.tolist()
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            probabilities = np.array([0.6, 0.4])
            return np.tile(probabilities, (len(X), 1))

    monkeypatch.setattr(
        impact,
        "build_xgboost_pipeline",
        lambda selected_feature_count=None: RecordingClassifier(),
    )

    results = impact.evaluate_filter_impact(
        features,
        samples,
    )

    assert len(fitted_models) == 2

    assert fitted_models[0]["training_rows"] == list(training_positions)
    assert fitted_models[1]["training_rows"] == list(training_positions[:-2])

    for fitted_model in fitted_models:
        assert fitted_model["validation_rows"] == list(validation_positions)

        assert not set(fitted_model["training_rows"]) & set(holdout_positions)

    fold = results["folds"][0]

    assert fold["removed_training_samples"] == 2
    assert fold["validation_samples"] == VALIDATION_COUNT
    assert results["holdout_samples"] == HOLDOUT_COUNT
    assert results["holdout_used"] is False
    assert results["validation_filtered"] is False

    assert fold["baseline"]["accuracy"] == pytest.approx(0.5)
    assert fold["filtered"]["accuracy"] == pytest.approx(0.5)
    assert fold["macro_f1_difference"] == pytest.approx(0.0)


def test_filter_impact_rejects_misaligned_rows():
    features, samples = make_test_data()

    with pytest.raises(ValueError, match="counts do not match"):
        impact.evaluate_filter_impact(
            features.iloc[:-1],
            samples,
        )
