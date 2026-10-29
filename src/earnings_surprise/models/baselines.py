import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from earnings_surprise.data.events import REQUIRED_PRIOR_EVENTS


PREVIOUS_OUTCOME_COLUMN = "beat_lag1"
BEAT_RATE_COLUMN = "beat_rate4"
BEAT_RATE_THRESHOLD = 0.5
BEAT_RATE_TOLERANCE = 1e-8


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


def _get_historical_beat_rates(features: pd.DataFrame) -> np.ndarray:
    """retrieves and validate four-event historical beat rates"""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("Features must be a pandas DataFrame")

    if BEAT_RATE_COLUMN not in features.columns:
        raise ValueError(
            f"Missing feature column: {BEAT_RATE_COLUMN}"
        )

    beat_rates = pd.to_numeric(
        features[BEAT_RATE_COLUMN],
        errors="coerce",
    ).to_numpy(dtype=float, na_value=np.nan)

    if len(beat_rates) == 0 or not np.isfinite(beat_rates).all():
        raise ValueError("beat_rate4 must contain finite values")

    if ((beat_rates < 0) | (beat_rates > 1)).any():
        raise ValueError("beat_rate4 must be between 0 and 1")

    # four historical events can produce five possible beat rates
    beat_counts = beat_rates * REQUIRED_PRIOR_EVENTS

    if not np.allclose(
        beat_counts,
        np.rint(beat_counts),
        atol=BEAT_RATE_TOLERANCE,
        rtol=0,
    ):
        raise ValueError("beat_rate4 contains an invalid proportion")

    return beat_rates


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
        

class AlwaysBeatBaseline(ClassifierMixin, BaseEstimator):
    """predicts an earnings beat for every observation"""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        if len(X) == 0:
            raise ValueError("Training features cannot be empty")

        if y is not None and len(X) != len(y):
            raise ValueError(
                "Training features and labels must have equal lengths"
            )

        self.classes_ = np.array([0, 1])

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "classes_")

        return np.ones(len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "classes_")

        return np.column_stack((
            np.zeros(len(X)),
            np.ones(len(X)),
        ))


class HistoricalBeatRateBaseline(ClassifierMixin, BaseEstimator):
    """predicts using earnings-beat frequency over four prior events"""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        _get_historical_beat_rates(X)

        if y is not None and len(X) != len(y):
            raise ValueError(
                "Training features and labels must have equal lengths"
            )

        self.classes_ = np.array([0, 1])

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "classes_")

        beat_rates = _get_historical_beat_rates(X)

        return (
            beat_rates >= BEAT_RATE_THRESHOLD
        ).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "classes_")

        beat_rates = _get_historical_beat_rates(X)

        return np.column_stack((
            1 - beat_rates,
            beat_rates,
        ))