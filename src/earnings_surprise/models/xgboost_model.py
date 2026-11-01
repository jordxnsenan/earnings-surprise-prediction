from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from earnings_surprise.preprocessing import build_preprocessor

XGB_TREE_COUNT = 200
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.05
XGB_MIN_CHILD_WEIGHT = 5
XGB_SUBSAMPLE = 0.8
XGB_COLUMN_SAMPLE = 0.8
XGB_RANDOM_STATE = 42
XGB_N_JOBS = 4


def build_xgboost_pipeline() -> Pipeline:
    """creates an XGBoost classifier with training-fitted preprocessing"""
    preprocessor = build_preprocessor()

    # no numerical scaling req. since Tree-based boosting
    preprocessor.set_params(continuous__scaler="passthrough")

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

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )