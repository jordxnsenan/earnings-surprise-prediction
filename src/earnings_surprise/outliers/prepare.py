import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import QuantileTransformer

from earnings_surprise.evaluation.splits import HOLDOUT_START
from earnings_surprise.features import FEATURE_COLUMNS
from earnings_surprise.preprocessing import build_preprocessor

OUTLIER_RANDOM_STATE = 42
OUTLIER_QUANTILES = 1000
OUTLIER_COMPARISON_SAMPLE_SIZE = 4000


def prepare_outlier_data(
    features: pd.DataFrame,
    samples: pd.DataFrame,
    sample_size: int | None = None,
    random_state: int = OUTLIER_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    """prepares development-only historical features for anomaly detection"""
    if len(features) != len(samples):
        raise ValueError("Feature and sample counts do not match")

    if not features.index.equals(samples.index):
        raise ValueError("Feature and sample indices do not match")

    if features.columns.tolist() != FEATURE_COLUMNS:
        raise ValueError("Feature columns do not match the specification")

    if "target_date_pub" not in samples.columns:
        raise ValueError("Missing target_date_pub column")

    dates = pd.to_datetime(
        samples["target_date_pub"],
        errors="raise",
    )

    if dates.isna().any():
        raise ValueError("Prediction dates cannot be missing")

    development_positions = np.flatnonzero((dates < HOLDOUT_START).to_numpy())

    if len(development_positions) < 2:
        raise ValueError("Not enough development observations")

    if sample_size is None:
        selected_positions = development_positions
    else:
        if (
            isinstance(sample_size, (bool, np.bool_))
            or not isinstance(sample_size, (int, np.integer))
            or sample_size < 2
            or sample_size > len(development_positions)
        ):
            raise ValueError("Invalid outlier sample size")

        generator = np.random.default_rng(random_state)

        selected_positions = np.sort(
            generator.choice(
                development_positions,
                size=sample_size,
                replace=False,
            )
        )

    selected_features = features.iloc[selected_positions].copy()

    preprocessor = build_preprocessor()

    preprocessor.set_params(
        continuous__scaler=QuantileTransformer(
            n_quantiles=min(
                OUTLIER_QUANTILES,
                len(selected_positions),
            ),
            output_distribution="normal",
            subsample=None,
            random_state=random_state,
        )
    )

    transformed = np.asarray(
        preprocessor.fit_transform(selected_features),
        dtype=float,
    )

    if not np.isfinite(transformed).all():
        raise ValueError("Outlier features contain non-finite values")

    return transformed, selected_positions, preprocessor