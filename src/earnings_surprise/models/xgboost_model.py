from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from earnings_surprise.preprocessing import (
    build_preprocessor,
    get_transformed_feature_names,
)
from earnings_surprise.selection import build_feature_selector

XGB_TREE_COUNT = 200
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.05
XGB_MIN_CHILD_WEIGHT = 5
XGB_SUBSAMPLE = 0.8
XGB_COLUMN_SAMPLE = 0.8
XGB_RANDOM_STATE = 42
XGB_N_JOBS = 4


def build_xgboost_pipeline(
    selected_feature_count: int | None = None,
) -> Pipeline:
    """creates an XGBoost classifier with training-fitted preprocessing"""
    preprocessor = build_preprocessor()

    # no numerical scaling req. since gradient boosted trees
    preprocessor.set_params(continuous__scaler="passthrough")

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

    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=XGB_TREE_COUNT,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        min_child_weight=XGB_MIN_CHILD_WEIGHT,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLUMN_SAMPLE,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=XGB_RANDOM_STATE,
        n_jobs=XGB_N_JOBS,
    )

    steps.append(("classifier", classifier))

    return Pipeline(steps)
