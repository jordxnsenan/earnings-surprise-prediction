import numpy as np
import pandas as pd

from earnings_surprise.features import FEATURE_COLUMNS
from earnings_surprise.preprocessing import (
    CATEGORICAL_COLUMNS,
    CONTINUOUS_COLUMNS,
    DISCRETE_COLUMNS,
    build_preprocessor,
    get_transformed_feature_names
)


def make_sample_features() -> pd.DataFrame:
    """creates synthetic historical features for preprocessing tests"""
    data = {
        column: [1.0, 2.0, 3.0]
        for column in CONTINUOUS_COLUMNS
    }

    for column in DISCRETE_COLUMNS:
        data[column] = [0, 1, 1]

    data["beat_rate4"] = [0.0, 0.5, 1.0]
    data["beat_streak4"] = [0, 1, 4]

    data["previous_report_month"] = pd.Categorical(
        [1, 2, 3],
        categories=range(1, 13),
    )

    data["previous_report_quarter"] = pd.Categorical(
        [1, 1, 1],
        categories=range(1, 5),
    )

    return pd.DataFrame(data)[FEATURE_COLUMNS]


def test_preprocessor_transforms_all_features():
    features = make_sample_features()
    original_features = features.copy(deep=True)

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(features)

    assert len(CONTINUOUS_COLUMNS) == 17
    assert len(DISCRETE_COLUMNS) == 6
    assert len(CATEGORICAL_COLUMNS) == 2

    assert transformed.shape == (3, 39)
    assert np.isfinite(transformed).all()

    transformed_names = preprocessor.get_feature_names_out()

    assert len(transformed_names) == 39
    assert "actual_eps_lag1" in transformed_names
    assert "previous_report_month_12" in transformed_names
    assert "previous_report_quarter_4" in transformed_names

    pd.testing.assert_frame_equal(features, original_features)


def test_preprocessor_uses_training_statistics():
    features = make_sample_features()

    training_features = features.copy()
    validation_features = features.iloc[[0]].copy()

    # inctroduces an extreme value in validation only
    validation_features.loc[
        validation_features.index[0], "actual_eps_lag1"
    ] = 1000.0

    preprocessor = build_preprocessor()
    preprocessor.fit(training_features)

    scaler = (
        preprocessor.named_transformers_["continuous"]
        .named_steps["scaler"]
    )

    actual_eps_index = CONTINUOUS_COLUMNS.index("actual_eps_lag1")

    # ensures scalers center comes from training data
    assert scaler.center_[actual_eps_index] == 2.0

    transformed_validation = preprocessor.transform(
        validation_features
    )

    assert transformed_validation.shape == (1, 39)

    # ensures transforming validation data doesnt change fitted statistics
    assert scaler.center_[actual_eps_index] == 2.0


def test_preprocessor_handles_missing_values():
    features = make_sample_features()

    features.loc[0, "actual_eps_lag1"] = np.nan
    features.loc[1, "beat_lag1"] = np.nan
    features.loc[2, "previous_report_month"] = np.nan

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(features)

    assert transformed.shape == (3, 39)
    assert np.isfinite(transformed).all()


def test_preprocessor_creates_independent_instances():
    first_preprocessor = build_preprocessor()
    second_preprocessor = build_preprocessor()

    assert first_preprocessor is not second_preprocessor

    # retrieves continuous pipelines before fitting
    first_continuous = first_preprocessor.transformers[0][1]
    second_continuous = second_preprocessor.transformers[0][1]

    assert first_continuous is not second_continuous

    # verifies that underlying transformers are independent
    assert first_continuous.named_steps["imputer"] is not (
        second_continuous.named_steps["imputer"]
    )

    assert first_continuous.named_steps["scaler"] is not (
        second_continuous.named_steps["scaler"]
    )
    

def test_transformed_feature_names_match_preprocessor():
    features = make_sample_features()

    preprocessor = build_preprocessor()
    preprocessor.fit(features)

    expected_names = get_transformed_feature_names()
    actual_names = list(preprocessor.get_feature_names_out())

    assert len(expected_names) == 39
    assert expected_names == actual_names