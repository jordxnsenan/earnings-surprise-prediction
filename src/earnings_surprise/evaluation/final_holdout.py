import numpy as np
import pandas as pd

from earnings_surprise.evaluation.ablation_features import FEATURE_SETS
from earnings_surprise.evaluation.metrics import evaluate_predictions
from earnings_surprise.features import FEATURE_COLUMNS
from earnings_surprise.models.baselines import (
    MajorityClassBaseline,
    PreviousOutcomeBaseline,
)
from earnings_surprise.models.xgboost_model import build_xgboost_pipeline
from earnings_surprise.evaluation.splits import (
    HOLDOUT_START,
    create_chronological_splits,
)

FINAL_FEATURE_SET = "without_trends"


def get_final_holdout_indices(
    samples: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """returns the development and reserved holdout row positions"""

    _, holdout_indices = create_chronological_splits(samples)

    all_indices = np.arange(len(samples))

    training_indices = np.setdiff1d(
        all_indices,
        holdout_indices,
    )

    training_dates = pd.to_datetime(samples.iloc[training_indices]["target_date_pub"])

    if not (training_dates < HOLDOUT_START).all():
        raise ValueError("Training data contains holdout observations")

    return training_indices, holdout_indices


def evaluate_final_holdout(
    features: pd.DataFrame,
    samples: pd.DataFrame,
) -> dict:
    """evaluates the selected model and baselines on the final holdout"""

    if len(features) != len(samples):
        raise ValueError("Feature and sample counts do not match")

    if not features.index.equals(samples.index):
        raise ValueError("Feature and sample indices do not match")

    if features.columns.tolist() != FEATURE_COLUMNS:
        raise ValueError("Feature columns do not match the specification")

    training_indices, holdout_indices = get_final_holdout_indices(samples)

    feature_columns = FEATURE_SETS[FINAL_FEATURE_SET]

    X_train = features.iloc[training_indices][feature_columns]
    X_holdout = features.iloc[holdout_indices][feature_columns]

    y_train = samples.iloc[training_indices]["target_beat"].astype(int)
    y_holdout = samples.iloc[holdout_indices]["target_beat"].astype(int)

    if y_train.nunique() != 2:
        raise ValueError("Training data must contain both classes")

    models = {
        "xgboost_without_trends": build_xgboost_pipeline(
            feature_columns=feature_columns
        ),
        "previous_outcome": PreviousOutcomeBaseline(),
        "majority_class": MajorityClassBaseline(),
    }

    results = {
        "experiment": "final_2024_holdout",
        "primary_metric": "macro_f1",
        "selected_feature_set": FINAL_FEATURE_SET,
        "feature_columns": feature_columns,
        "training_samples": len(training_indices),
        "holdout_samples": len(holdout_indices),
        "models": {},
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_holdout)
        probabilities = model.predict_proba(X_holdout)[:, 1]

        metrics = evaluate_predictions(
            y_holdout,
            predictions,
            probabilities,
        )

        results["models"][name] = metrics

    return results
