import numpy as np
import pandas as pd

from sklearn.preprocessing import QuantileTransformer

from earnings_surprise.evaluation.splits import HOLDOUT_START
from earnings_surprise.features import FEATURE_COLUMNS
from earnings_surprise.preprocessing import build_preprocessor

CLUSTER_SAMPLE_SIZE = 4000
CLUSTER_RANDOM_STATE = 42
CLUSTER_QUANTILES = 1000


def prepare_clustering_data(
    features: pd.DataFrame,
    samples: pd.DataFrame,
    sample_size: int = CLUSTER_SAMPLE_SIZE,
    random_state: int = CLUSTER_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, object]:
    """samples and preprocesses development observations"""
    if len(features) != len(samples):
        raise ValueError("Feature and sample counts do not match")

    if not features.index.equals(samples.index):
        raise ValueError("Feature and sample indices do not match")

    if features.columns.tolist() != FEATURE_COLUMNS:
        raise ValueError("Feature columns do not match the specification")

    if "target_date_pub" not in samples.columns:
        raise ValueError("Missing target_date_pub column")

    if isinstance(sample_size, bool) or sample_size < 2:
        raise ValueError("Sample size must be at least 2")

    dates = pd.to_datetime(
        samples["target_date_pub"],
        errors="raise",
    )

    if dates.isna().any():
        raise ValueError("Prediction dates cannot be missing")

    # keeps final holdout outside the clustering population
    development_mask = dates < HOLDOUT_START
    development_positions = np.flatnonzero(development_mask.to_numpy())

    if len(development_positions) < 2:
        raise ValueError("Not enough development observations")

    selected_count = min(
        sample_size,
        len(development_positions),
    )

    generator = np.random.default_rng(random_state)

    selected_positions = np.sort(
        generator.choice(
            development_positions,
            size=selected_count,
            replace=False,
        )
    )

    selected_features = features.iloc[selected_positions].copy()

    preprocessor = build_preprocessor()

    preprocessor.set_params(
        continuous__scaler=QuantileTransformer(
            n_quantiles=min(CLUSTER_QUANTILES, selected_count),
            output_distribution="normal",
            subsample=None,
            random_state=random_state,
        )
    )

    transformed = preprocessor.fit_transform(selected_features)
    transformed = np.asarray(transformed, dtype=float)

    if not np.isfinite(transformed).all():
        raise ValueError("Clustering data contains non-finite values")

    return transformed, selected_positions, preprocessor