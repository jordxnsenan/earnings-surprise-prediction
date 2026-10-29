import numpy as np
import pandas as pd
import pytest

from earnings_surprise.features import FEATURE_COLUMNS
from earnings_surprise.preprocessing import (
    CATEGORICAL_COLUMNS,
    CONTINUOUS_COLUMNS,
    DISCRETE_COLUMNS,
    build_preprocessor,
)
from earnings_surprise.selection import (
    DEFAULT_SELECTED_FEATURES,
    build_feature_selector,
    get_selected_feature_names,
)


def make_sample_features() -> pd.DataFrame:
    """creates synthetic historical features for selector tests"""
    row_count = 20

    data = {}

    for column in CONTINUOUS_COLUMNS:
        data[column] = np.linspace(0.0, 1.0, row_count)

    for column in DISCRETE_COLUMNS:
        data[column] = [0, 1] * 10

    # gives stronger signals to a few features
    data["actual_eps_lag1"] = np.arange(row_count, dtype=float)
    data["surprise_eps_lag1"] = np.arange(row_count, dtype=float)
    data["beat_rate4"] = [0, 1] * 10
    data["beat_streak4"] = [0, 1] * 10

    data["previous_report_month"] = pd.Categorical(
        [(i % 12) + 1 for i in range(row_count)],
        categories=range(1, 13),
    )

    data["previous_report_quarter"] = pd.Categorical(
        [(i % 4) + 1 for i in range(row_count)],
        categories=range(1, 5),
    )

    return pd.DataFrame(data)[FEATURE_COLUMNS]


def make_sample_targets() -> pd.Series:
    """creates a simple binary target"""
    return pd.Series([0, 1] * 10, name="target_beat")


def test_feature_selector_keeps_expected_number_of_features():
    features = make_sample_features()
    targets = make_sample_targets()

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(features)
    transformed_names = preprocessor.get_feature_names_out()

    transformed_names = preprocessor.get_feature_names_out()
    selector = build_feature_selector(transformed_names)
    
    selected = selector.fit_transform(transformed, targets)

    selected_names = get_selected_feature_names(
        transformed_names,
        selector,
    )

    assert transformed.shape == (20, 39)
    assert selected.shape == (20, DEFAULT_SELECTED_FEATURES)
    assert len(selected_names) == DEFAULT_SELECTED_FEATURES
    assert len(set(selected_names)) == DEFAULT_SELECTED_FEATURES


def test_feature_selector_uses_training_fit_only():
    features = make_sample_features()
    targets = make_sample_targets()

    training_features = features.iloc[:16].copy()
    training_targets = targets.iloc[:16].copy()

    validation_features = features.iloc[16:].copy()

    preprocessor = build_preprocessor()

    transformed_train = preprocessor.fit_transform(training_features)
    transformed_validation = preprocessor.transform(validation_features)

    transformed_names = preprocessor.get_feature_names_out()
    selector = build_feature_selector(transformed_names)
    
    selector.fit(transformed_train, training_targets)

    selected_names_before = get_selected_feature_names(
        preprocessor.get_feature_names_out(),
        selector,
    )

    transformed_validation[0, 0] = 9999.0
    _ = selector.transform(transformed_validation)

    selected_names_after = get_selected_feature_names(
        preprocessor.get_feature_names_out(),
        selector,
    )

    assert selected_names_before == selected_names_after


def test_feature_selector_instances_are_independent():
    features = make_sample_features()

    preprocessor = build_preprocessor()
    preprocessor.fit(features)

    transformed_names = preprocessor.get_feature_names_out()

    first_selector = build_feature_selector(transformed_names)
    second_selector = build_feature_selector(transformed_names)

    assert first_selector is not second_selector
    assert first_selector.score_func is not second_selector.score_func


def test_feature_selector_rejects_invalid_k():
    features = make_sample_features()

    preprocessor = build_preprocessor()
    preprocessor.fit(features)

    transformed_names = preprocessor.get_feature_names_out()

    with pytest.raises(ValueError, match="positive integer"):
        build_feature_selector(transformed_names, k=0)

    with pytest.raises(ValueError, match="positive integer"):
        build_feature_selector(transformed_names, k=-1)

    with pytest.raises(ValueError, match="positive integer"):
        build_feature_selector(transformed_names, k=True)


def test_selected_feature_names_require_matching_lengths():
    features = make_sample_features()
    targets = make_sample_targets()

    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(features)
    transformed_names = preprocessor.get_feature_names_out()

    selector = build_feature_selector(transformed_names)
    selector.fit(transformed, targets)

    # intentionally provides fewer names than the selector expects
    incomplete_names = transformed_names[:-1]

    with pytest.raises(
        ValueError,
        match="Feature-name count does not match",
    ):
        get_selected_feature_names(incomplete_names, selector)


def test_mutual_information_uses_correct_feature_types(monkeypatch):
    features = make_sample_features()
    targets = make_sample_targets()

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(features)

    transformed_names = list(
        preprocessor.get_feature_names_out()
    )

    original_transformed = transformed.copy()
    captured = {}

    def fake_mutual_information(
        X,
        y,
        *,
        discrete_features,
        random_state,
    ):
        captured["data"] = X.copy()
        captured["discrete_mask"] = discrete_features.copy()

        return np.arange(X.shape[1], dtype=float)

    monkeypatch.setattr(
        "earnings_surprise.selection.mutual_info_classif",
        fake_mutual_information,
    )

    selector = build_feature_selector(transformed_names)
    selector.fit(transformed, targets)

    beat_rate_index = transformed_names.index("beat_rate4")
    actual_eps_index = transformed_names.index("actual_eps_lag1")
    month_index = transformed_names.index("previous_report_month_1")

    discrete_mask = captured["discrete_mask"]

    assert discrete_mask.sum() == 22
    assert discrete_mask[beat_rate_index]
    assert discrete_mask[month_index]
    assert not discrete_mask[actual_eps_index]

    expected_beat_counts = np.rint(
        transformed[:, beat_rate_index] * 4
    )

    np.testing.assert_array_equal(
        captured["data"][:, beat_rate_index],
        expected_beat_counts,
    )

    # ensure mutual information scoring doesnt modify feature matrix
    np.testing.assert_array_equal(
        transformed,
        original_transformed,
    )