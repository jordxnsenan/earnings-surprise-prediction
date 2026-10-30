from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from earnings_surprise.preprocessing import (
    build_preprocessor,
    get_transformed_feature_names,
)
from earnings_surprise.selection import build_feature_selector


LOGISTIC_MAX_ITERATIONS = 2000
LOGISTIC_REGULARIZATION_STRENGTH = 1.0
LOGISTIC_RANDOM_STATE = 42

QUANTILE_COUNT = 1000

ROBUST_SCALER = "robust"
QUANTILE_SCALER = "quantile"


def build_logistic_regression_pipeline(
    continuous_scaler: str = ROBUST_SCALER,
    selected_feature_count: int | None = None,
) -> Pipeline:
    """creates Logistic Regression with optional feature selection"""
    if continuous_scaler not in (ROBUST_SCALER, QUANTILE_SCALER):
        raise ValueError(
            f"Unknown continuous scaler: {continuous_scaler}"
        )

    preprocessor = build_preprocessor()

    if continuous_scaler == QUANTILE_SCALER:
        preprocessor.set_params(
            continuous__scaler=QuantileTransformer(
                n_quantiles=QUANTILE_COUNT,
                output_distribution="normal",
                subsample=None,
                random_state=LOGISTIC_RANDOM_STATE,
            )
        )

    steps = [
        ("preprocessor", preprocessor),
    ]

    if selected_feature_count is not None:
        transformed_names = get_transformed_feature_names()

        selector = build_feature_selector(
            transformed_names,
            k=selected_feature_count,
        )

        steps.append(("selector", selector))

    classifier = LogisticRegression(
        C=LOGISTIC_REGULARIZATION_STRENGTH,
        solver="lbfgs",
        max_iter=LOGISTIC_MAX_ITERATIONS,
        random_state=LOGISTIC_RANDOM_STATE,
    )

    steps.append(("classifier", classifier))

    return Pipeline(steps)