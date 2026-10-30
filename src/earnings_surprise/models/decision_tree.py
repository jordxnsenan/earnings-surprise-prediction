from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from earnings_surprise.preprocessing import build_preprocessor


TREE_MAX_DEPTH = 6
TREE_MIN_SAMPLES_LEAF = 10
TREE_RANDOM_STATE = 42


def build_decision_tree_pipeline() -> Pipeline:
    """creates a Decision Tree with training-fitted preprocessing"""
    preprocessor = build_preprocessor()

    # trees do not require numerical feature scaling
    preprocessor.set_params(
        continuous__scaler="passthrough"
    )

    classifier = DecisionTreeClassifier(
        criterion="gini",
        max_depth=TREE_MAX_DEPTH,
        min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
        random_state=TREE_RANDOM_STATE,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])