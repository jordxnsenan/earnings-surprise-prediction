from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from earnings_surprise.selection import build_feature_selector
from earnings_surprise.preprocessing import (
    build_preprocessor,
    get_transformed_feature_names,
)


TREE_MAX_DEPTH = 6
TREE_MIN_SAMPLES_LEAF = 10
TREE_RANDOM_STATE = 42


def build_decision_tree_pipeline(
    selected_feature_count: int | None = None,
) -> Pipeline:
    """creates a Decision Tree with optional feature selection"""
    preprocessor = build_preprocessor()

    # decision trees don't require numerical scaling
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

    classifier = DecisionTreeClassifier(
        criterion="gini",
        max_depth=TREE_MAX_DEPTH,
        min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
        random_state=TREE_RANDOM_STATE,
    )

    steps.append(("classifier", classifier))

    return Pipeline(steps)