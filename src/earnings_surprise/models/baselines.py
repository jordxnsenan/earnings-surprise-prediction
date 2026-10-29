import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


PREVIOUS_OUTCOME_COLUMN = "beat_lag1"


def _validate_binary_values(values, description: str) -> np.ndarray:
    """validates binary values without replacing missing observations"""
    binary_values = pd.Series(values)

    if binary_values.empty:
        raise ValueError(f"{description} cannot be empty")

    if binary_values.isna().any() or not binary_values.isin([0, 1]).all():
        raise ValueError(f"{description} must contain only 0 or 1")

    return binary_values.to_numpy(dtype=int)


def _get_previous_outcomes(features: pd.DataFrame) -> np.ndarray:
    """retrieves the most recent historical earnings outcomes"""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("Features must be a pandas DataFrame")

    if PREVIOUS_OUTCOME_COLUMN not in features.columns:
        raise ValueError(
            f"Missing feature column: {PREVIOUS_OUTCOME_COLUMN}"
        )

    return _validate_binary_values(
        features[PREVIOUS_OUTCOME_COLUMN],
        PREVIOUS_OUTCOME_COLUMN,
    )


class MajorityClassBaseline(ClassifierMixin, BaseEstimator):
    """predicts the most frequent class in the training population"""

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if len(X) != len(y):
            raise ValueError(
                "Training features and labels must have equal lengths"
            )

        labels = _validate_binary_values(y, "Training labels")
        class_counts = np.bincount(labels, minlength=2)

        # equal class counts are resolved in favour of class 0
        self.majority_class_ = int(np.argmax(class_counts))
        self.beat_probability_ = float(class_counts[1] / len(labels))
        self.classes_ = np.array([0, 1])

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "majority_class_")

        return np.full(
            len(X),
            self.majority_class_,
            dtype=int,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "beat_probability_")

        probabilities = [
            1 - self.beat_probability_,
            self.beat_probability_,
        ]

        return np.tile(probabilities, (len(X), 1))


class PreviousOutcomeBaseline(ClassifierMixin, BaseEstimator):
    """repeats the outcome of the most recent historical earnings event"""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        _get_previous_outcomes(X)

        if y is not None and len(X) != len(y):
            raise ValueError(
                "Training features and labels must have equal lengths"
            )

        self.classes_ = np.array([0, 1])

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "classes_")

        return _get_previous_outcomes(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        previous_outcomes = self.predict(X).astype(float)

        return np.column_stack((
            1 - previous_outcomes,
            previous_outcomes,
        ))