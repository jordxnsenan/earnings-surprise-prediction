import numpy as np

from functools import partial
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from earnings_surprise.data.events import REQUIRED_PRIOR_EVENTS
from earnings_surprise.preprocessing import (
    CATEGORICAL_COLUMNS,
    CONTINUOUS_COLUMNS,
    DISCRETE_COLUMNS,
)


DEFAULT_SELECTED_FEATURES = 15
MUTUAL_INFORMATION_RANDOM_STATE = 42
BEAT_RATE_FEATURE = "beat_rate4"


def is_categorical_indicator(feature_name: str) -> bool:
    """checks whether a column is an encoded calendar feature"""
    return any(
        feature_name.startswith(f"{column}_")
        for column in CATEGORICAL_COLUMNS
    )


def calculate_mutual_information(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    """calculates mutual information using the feature types"""
    scoring_data = np.asarray(X, dtype=float).copy()

    if scoring_data.ndim != 2 or scoring_data.shape[1] != len(feature_names):
        raise ValueError("Feature names do not match input columns")

    if not np.isfinite(scoring_data).all():
        raise ValueError("Feature matrix contains non-finite values")

    known_features = set(CONTINUOUS_COLUMNS + DISCRETE_COLUMNS)

    unknown_features = [
        name for name in feature_names
        if name not in known_features
        and not is_categorical_indicator(name)
    ]

    if unknown_features:
        raise ValueError(
            f"Unknown transformed features: {unknown_features}"
        )

    discrete_feature_mask = np.array([
        name in DISCRETE_COLUMNS or is_categorical_indicator(name)
        for name in feature_names
    ], dtype=bool)

    # represents beat-rate categories as integer beat counts when estimating mutual information
    beat_rate_index = feature_names.index(BEAT_RATE_FEATURE)

    beat_counts = (
        scoring_data[:, beat_rate_index] * REQUIRED_PRIOR_EVENTS
    )

    rounded_counts = np.rint(beat_counts)

    if (
        not np.allclose(beat_counts, rounded_counts)
        or (rounded_counts < 0).any()
        or (rounded_counts > REQUIRED_PRIOR_EVENTS).any()
    ):
        raise ValueError("Unexpected beat_rate4 values")

    scoring_data[:, beat_rate_index] = rounded_counts

    return mutual_info_classif(
        scoring_data,
        y,
        discrete_features=discrete_feature_mask,
        random_state=MUTUAL_INFORMATION_RANDOM_STATE,
    )


def build_feature_selector(
    transformed_feature_names: list[str] | np.ndarray,
    k: int = DEFAULT_SELECTED_FEATURES,
) -> SelectKBest:
    """creates feature selector using the transformed feature names"""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    feature_names = tuple(transformed_feature_names)

    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Duplicate transformed feature names")

    if BEAT_RATE_FEATURE not in feature_names:
        raise ValueError("Missing beat_rate4 feature")

    if k > len(feature_names):
        raise ValueError("k exceeds the available feature count")

    score_function = partial(
        calculate_mutual_information,
        feature_names=feature_names,
    )

    return SelectKBest(
        score_func=score_function,
        k=k,
    )


def get_selected_feature_names(
    transformed_feature_names: list[str] | np.ndarray,
    selector: SelectKBest,
) -> list[str]:
    """returns selected feature names in original column order"""
    names = np.asarray(transformed_feature_names)
    support_mask = selector.get_support()

    if len(names) != len(support_mask):
        raise ValueError(
            "Feature-name count does not match selector support mask"
        )

    return names[support_mask].tolist()