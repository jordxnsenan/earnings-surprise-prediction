import numpy as np
import pandas as pd
import pytest

from earnings_surprise.models.baselines import (
    MajorityClassBaseline,
    PreviousOutcomeBaseline,
)


def test_majority_class_baseline_uses_training_labels():
    training = pd.DataFrame({
        "beat_lag1": [0, 1, 0, 1, 0],
    })

    labels = pd.Series([1, 1, 0, 1, 0], dtype="Int8")

    validation = pd.DataFrame({
        "beat_lag1": [0, 0, 1],
    })

    model = MajorityClassBaseline()
    model.fit(training, labels)

    predictions = model.predict(validation)
    probabilities = model.predict_proba(validation)

    assert predictions.tolist() == [1, 1, 1]

    np.testing.assert_allclose(
        probabilities[:, 1],
        [0.6, 0.6, 0.6],
    )


def test_majority_class_baseline_tie_uses_class_zero():
    training = pd.DataFrame({
        "beat_lag1": [0, 1, 1, 0],
    })

    labels = pd.Series([1, 0, 1, 0])

    model = MajorityClassBaseline()
    model.fit(training, labels)

    assert model.predict(training).tolist() == [0, 0, 0, 0]

    np.testing.assert_allclose(
        model.predict_proba(training)[:, 1],
        [0.5, 0.5, 0.5, 0.5],
    )


def test_majority_class_baseline_rejects_invalid_labels():
    training = pd.DataFrame({
        "beat_lag1": [1, 0],
    })

    labels = pd.Series([1, pd.NA], dtype="Int8")

    with pytest.raises(ValueError, match="only 0 or 1"):
        MajorityClassBaseline().fit(training, labels)


def test_previous_outcome_baseline_follows_lag_one():
    training = pd.DataFrame({
        "beat_lag1": [0, 1, 1],
    })

    validation = pd.DataFrame({
        "beat_lag1": pd.Series([1, 0, 1], dtype="Int8"),
    })

    model = PreviousOutcomeBaseline()
    model.fit(training)

    predictions = model.predict(validation)
    probabilities = model.predict_proba(validation)

    assert predictions.tolist() == [1, 0, 1]

    np.testing.assert_array_equal(
        probabilities,
        np.array([
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]),
    )


def test_previous_outcome_baseline_rejects_missing_values():
    training = pd.DataFrame({
        "beat_lag1": [0, 1],
    })

    model = PreviousOutcomeBaseline()
    model.fit(training)

    validation = pd.DataFrame({
        "beat_lag1": pd.Series([1, pd.NA], dtype="Int8"),
    })

    with pytest.raises(ValueError, match="only 0 or 1"):
        model.predict(validation)


def test_previous_outcome_baseline_requires_feature_column():
    training = pd.DataFrame({
        "other_feature": [0, 1],
    })

    with pytest.raises(ValueError, match="Missing feature column"):
        PreviousOutcomeBaseline().fit(training)