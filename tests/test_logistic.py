import numpy as np
import pandas as pd
import pytest

from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from earnings_surprise.features import FEATURE_COLUMNS
from earnings_surprise.models.logistic import (
    LOGISTIC_MAX_ITERATIONS,
    build_logistic_regression_pipeline,
)
from earnings_surprise.preprocessing import (
    CONTINUOUS_COLUMNS,
    DISCRETE_COLUMNS,
)


TRAINING_SAMPLE_COUNT = 32
VALIDATION_SAMPLE_COUNT = 8
TOTAL_SAMPLE_COUNT = TRAINING_SAMPLE_COUNT + VALIDATION_SAMPLE_COUNT


def make_sample_data():
    """creates synthetic historical features and binary labels"""
    data = {
        column: np.linspace(-2.0, 2.0, TOTAL_SAMPLE_COUNT)
        for column in CONTINUOUS_COLUMNS
    }

    for column in DISCRETE_COLUMNS:
        data[column] = [
            index % 2 for index in range(TOTAL_SAMPLE_COUNT)
        ]

    data["beat_rate4"] = [
        float(index % 2) for index in range(TOTAL_SAMPLE_COUNT)
    ]

    data["beat_streak4"] = [
        4 if index % 2 else 0
        for index in range(TOTAL_SAMPLE_COUNT)
    ]

    months = [
        (index % 12) + 1
        for index in range(TOTAL_SAMPLE_COUNT)
    ]

    data["previous_report_month"] = pd.Categorical(
        months,
        categories=range(1, 13),
    )

    data["previous_report_quarter"] = pd.Categorical(
        [(month - 1) // 3 + 1 for month in months],
        categories=range(1, 5),
    )

    features = pd.DataFrame(data)[FEATURE_COLUMNS]

    targets = pd.Series(
        [index % 2 for index in range(TOTAL_SAMPLE_COUNT)],
        dtype="int8",
    )

    return features, targets


def test_logistic_pipeline_construction():
    model = build_logistic_regression_pipeline()

    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["classifier"], LogisticRegression)

    assert model.named_steps["classifier"].solver == "lbfgs"
    assert model.named_steps["classifier"].max_iter == (
        LOGISTIC_MAX_ITERATIONS
    )


def test_logistic_pipeline_fits_and_predicts():
    features, targets = make_sample_data()

    X_train = features.iloc[:TRAINING_SAMPLE_COUNT]
    X_validation = features.iloc[TRAINING_SAMPLE_COUNT:]

    y_train = targets.iloc[:TRAINING_SAMPLE_COUNT]

    model = build_logistic_regression_pipeline()
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

    transformed_names = (
        model.named_steps["preprocessor"].get_feature_names_out()
    )

    assert len(transformed_names) == 39


def test_logistic_pipeline_uses_training_statistics():
    features, targets = make_sample_data()

    X_train = features.iloc[:TRAINING_SAMPLE_COUNT]
    y_train = targets.iloc[:TRAINING_SAMPLE_COUNT]

    X_validation = features.iloc[TRAINING_SAMPLE_COUNT:].copy()

    # introduces an extreme value in validation only
    X_validation.loc[
        X_validation.index[0], "actual_eps_lag1"
    ] = 1_000_000.0

    model = build_logistic_regression_pipeline()
    model.fit(X_train, y_train)

    scaler = (
        model.named_steps["preprocessor"]
        .named_transformers_["continuous"]
        .named_steps["scaler"]
    )

    actual_eps_index = CONTINUOUS_COLUMNS.index("actual_eps_lag1")

    training_median = X_train["actual_eps_lag1"].median()

    assert np.isclose(
        scaler.center_[actual_eps_index],
        training_median,
    )

    original_centers = scaler.center_.copy()

    # ensures predictions dont refit the scaler
    model.predict_proba(X_validation)

    np.testing.assert_array_equal(
        scaler.center_,
        original_centers,
    )


def test_logistic_pipeline_instances_are_independent():
    first_model = build_logistic_regression_pipeline()
    second_model = build_logistic_regression_pipeline()

    assert first_model is not second_model

    assert first_model.named_steps["preprocessor"] is not (
        second_model.named_steps["preprocessor"]
    )

    assert first_model.named_steps["classifier"] is not (
        second_model.named_steps["classifier"]
    )
    

def test_logistic_pipeline_supports_scaling_options():
    robust_model = build_logistic_regression_pipeline(
        continuous_scaler="robust"
    )

    quantile_model = build_logistic_regression_pipeline(
        continuous_scaler="quantile"
    )

    robust_scaler = (
        robust_model.named_steps["preprocessor"]
        .transformers[0][1]
        .named_steps["scaler"]
    )

    quantile_scaler = (
        quantile_model.named_steps["preprocessor"]
        .transformers[0][1]
        .named_steps["scaler"]
    )

    assert isinstance(robust_scaler, RobustScaler)
    assert isinstance(quantile_scaler, QuantileTransformer)

    assert quantile_scaler.output_distribution == "normal"
    assert quantile_scaler.subsample is None


def test_quantile_pipeline_uses_training_data_only():
    features, targets = make_sample_data()

    X_train = features.iloc[:TRAINING_SAMPLE_COUNT]
    y_train = targets.iloc[:TRAINING_SAMPLE_COUNT]

    X_validation = features.iloc[TRAINING_SAMPLE_COUNT:].copy()

    model = build_logistic_regression_pipeline(
        continuous_scaler="quantile"
    )

    # uses fewer quantiles for this dataset
    model.set_params(
        preprocessor__continuous__scaler__n_quantiles=10
    )

    model.fit(X_train, y_train)

    quantile_scaler = (
        model.named_steps["preprocessor"]
        .named_transformers_["continuous"]
        .named_steps["scaler"]
    )

    original_quantiles = quantile_scaler.quantiles_.copy()

    # ensures extreme validation values dont change fitted quantiles
    X_validation.loc[
        X_validation.index[0], "actual_eps_lag1"
    ] = 1_000_000.0

    probabilities = model.predict_proba(X_validation)

    assert probabilities.shape == (VALIDATION_SAMPLE_COUNT, 2)
    assert np.isfinite(probabilities).all()

    np.testing.assert_array_equal(
        quantile_scaler.quantiles_,
        original_quantiles,
    )


def test_logistic_pipeline_rejects_unknown_scaler():
    with pytest.raises(ValueError, match="Unknown continuous scaler"):
        build_logistic_regression_pipeline(
            continuous_scaler="unsupported"
        )