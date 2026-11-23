import numpy as np
import pandas as pd
import pytest

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from earnings_surprise.selection import get_selected_feature_names
from earnings_surprise.features import FEATURE_COLUMNS
from earnings_surprise.models.decision_tree import (
    TREE_MAX_DEPTH,
    TREE_MIN_SAMPLES_LEAF,
    build_decision_tree_pipeline,
)
from earnings_surprise.preprocessing import (
    CONTINUOUS_COLUMNS,
    DISCRETE_COLUMNS,
)


TOTAL_SAMPLE_COUNT = 48
TRAINING_SAMPLE_COUNT = 36
VALIDATION_SAMPLE_COUNT = TOTAL_SAMPLE_COUNT - TRAINING_SAMPLE_COUNT


def make_sample_data():
    """Create synthetic historical features and binary targets."""
    positions = np.arange(TOTAL_SAMPLE_COUNT)

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

    streaks = np.zeros(TOTAL_SAMPLE_COUNT, dtype=int)
    consecutive = np.ones(TOTAL_SAMPLE_COUNT, dtype=bool)

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

    targets = pd.Series(
        (positions // 3) % 2,
        dtype="int8",
    )

    return features, targets


def test_decision_tree_pipeline_construction():
    model = build_decision_tree_pipeline()

    assert isinstance(model, Pipeline)

    classifier = model.named_steps["classifier"]

    assert isinstance(classifier, DecisionTreeClassifier)
    assert classifier.max_depth == TREE_MAX_DEPTH
    assert classifier.min_samples_leaf == TREE_MIN_SAMPLES_LEAF

    continuous_pipeline = (
        model.named_steps["preprocessor"]
        .transformers[0][1]
    )

    assert continuous_pipeline.named_steps["scaler"] == "passthrough"


def test_decision_tree_fits_and_predicts():
    features, targets = make_sample_data()

    original_features = features.copy(deep=True)

    X_train = features.iloc[:TRAINING_SAMPLE_COUNT]
    X_validation = features.iloc[TRAINING_SAMPLE_COUNT:]

    y_train = targets.iloc[:TRAINING_SAMPLE_COUNT]

    model = build_decision_tree_pipeline()
    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)
    probabilities = model.predict_proba(X_validation)

    assert predictions.shape == (VALIDATION_SAMPLE_COUNT,)
    assert probabilities.shape == (VALIDATION_SAMPLE_COUNT, 2)

    assert np.isin(predictions, [0, 1]).all()
    assert np.isfinite(probabilities).all()

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(VALIDATION_SAMPLE_COUNT),
    )

    classifier = model.named_steps["classifier"]

    assert classifier.get_depth() <= TREE_MAX_DEPTH
    assert classifier.n_features_in_ == 39

    pd.testing.assert_frame_equal(features, original_features)


def test_decision_tree_uses_training_statistics():
    features, targets = make_sample_data()

    X_train = features.iloc[:TRAINING_SAMPLE_COUNT].copy()
    y_train = targets.iloc[:TRAINING_SAMPLE_COUNT]

    X_train.loc[0, "actual_eps_lag1"] = np.nan

    model = build_decision_tree_pipeline()
    model.fit(X_train, y_train)

    continuous_pipeline = (
        model.named_steps["preprocessor"]
        .named_transformers_["continuous"]
    )

    imputer = continuous_pipeline.named_steps["imputer"]

    feature_index = CONTINUOUS_COLUMNS.index("actual_eps_lag1")

    training_median = X_train["actual_eps_lag1"].median()

    assert np.isclose(
        imputer.statistics_[feature_index],
        training_median,
    )

    original_statistics = imputer.statistics_.copy()

    X_validation = features.iloc[TRAINING_SAMPLE_COUNT:].copy()

    X_validation.loc[
        X_validation.index[0], "actual_eps_lag1"
    ] = 1_000_000.0

    model.predict(X_validation)

    np.testing.assert_array_equal(
        imputer.statistics_,
        original_statistics,
    )


def test_decision_tree_predictions_are_reproducible():
    features, targets = make_sample_data()

    X_train = features.iloc[:TRAINING_SAMPLE_COUNT]
    X_validation = features.iloc[TRAINING_SAMPLE_COUNT:]

    y_train = targets.iloc[:TRAINING_SAMPLE_COUNT]

    first_model = build_decision_tree_pipeline()
    second_model = build_decision_tree_pipeline()

    first_model.fit(X_train, y_train)
    second_model.fit(X_train, y_train)

    np.testing.assert_array_equal(
        first_model.predict(X_validation),
        second_model.predict(X_validation),
    )

    np.testing.assert_allclose(
        first_model.predict_proba(X_validation),
        second_model.predict_proba(X_validation),
    )


def test_decision_tree_with_feature_selection():
    features, targets = make_sample_data()

    X_train = features.iloc[:TRAINING_SAMPLE_COUNT]
    X_validation = features.iloc[TRAINING_SAMPLE_COUNT:]

    y_train = targets.iloc[:TRAINING_SAMPLE_COUNT]

    model = build_decision_tree_pipeline(
        selected_feature_count=15
    )

    model.fit(X_train, y_train)

    preprocessor = model.named_steps["preprocessor"]
    selector = model.named_steps["selector"]
    classifier = model.named_steps["classifier"]

    transformed_names = preprocessor.get_feature_names_out()

    selected_names = get_selected_feature_names(
        transformed_names,
        selector,
    )

    assert len(transformed_names) == 39
    assert selector.get_support().sum() == 15
    assert len(selected_names) == 15

    assert classifier.n_features_in_ == 15
    assert classifier.get_depth() <= TREE_MAX_DEPTH

    predictions = model.predict(X_validation)
    probabilities = model.predict_proba(X_validation)

    assert predictions.shape == (VALIDATION_SAMPLE_COUNT,)
    assert probabilities.shape == (VALIDATION_SAMPLE_COUNT, 2)

    assert np.isin(predictions, [0, 1]).all()
    assert np.isfinite(probabilities).all()

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(VALIDATION_SAMPLE_COUNT),
    )


def test_decision_tree_selection_does_not_refit_on_validation():
    features, targets = make_sample_data()

    X_train = features.iloc[:TRAINING_SAMPLE_COUNT]
    X_validation = features.iloc[TRAINING_SAMPLE_COUNT:].copy()

    y_train = targets.iloc[:TRAINING_SAMPLE_COUNT]

    model = build_decision_tree_pipeline(
        selected_feature_count=15
    )

    model.fit(X_train, y_train)

    selector = model.named_steps["selector"]

    original_support = selector.get_support().copy()
    original_scores = selector.scores_.copy()

    # modifies validation data after the model has been fitted
    X_validation.loc[
        X_validation.index[0], "actual_eps_lag1"
    ] = 1_000_000.0

    model.predict(X_validation)

    np.testing.assert_array_equal(
        selector.get_support(),
        original_support,
    )

    np.testing.assert_array_equal(
        selector.scores_,
        original_scores,
    )


def test_decision_tree_rejects_invalid_feature_count():
    with pytest.raises(ValueError, match="positive integer"):
        build_decision_tree_pipeline(
            selected_feature_count=0
        )