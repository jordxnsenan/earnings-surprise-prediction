from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from earnings_surprise.preprocessing import (
    build_preprocessor,
    get_transformed_feature_names,
)
from earnings_surprise.selection import build_feature_selector


FOREST_TREE_COUNT = 200
FOREST_MAX_DEPTH = 10
FOREST_MIN_SAMPLES_LEAF = 10
FOREST_MAX_FEATURES = "sqrt"
FOREST_RANDOM_STATE = 42


def build_random_forest_pipeline(
    selected_feature_count: int | None = None,
) -> Pipeline:
    """creates a Random Forest with training-fitted preprocessing"""
    preprocessor = build_preprocessor()

    # numerical scaling not req. since its a tree-based model
    preprocessor.set_params(
        continuous__scaler="passthrough"
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

    classifier = RandomForestClassifier(
        n_estimators=FOREST_TREE_COUNT,
        criterion="gini",
        max_depth=FOREST_MAX_DEPTH,
        min_samples_leaf=FOREST_MIN_SAMPLES_LEAF,
        max_features=FOREST_MAX_FEATURES,
        bootstrap=True,
        random_state=FOREST_RANDOM_STATE,
        n_jobs=-1,
    )

    steps.append(("classifier", classifier))

    return Pipeline(steps)