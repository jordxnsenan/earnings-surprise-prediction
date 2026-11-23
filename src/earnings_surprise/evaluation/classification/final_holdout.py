import json
import numpy as np
import pandas as pd

from pathlib import Path
from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.ablation_features import FEATURE_SETS
from earnings_surprise.evaluation.metrics import evaluate_predictions
from earnings_surprise.features import FEATURE_COLUMNS, FEATURE_MATRIX_PATH
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

FINAL_RESULTS_PATH = Path("results/final_holdout/final_holdout_results.json")
FINAL_PREDICTIONS_PATH = Path("results/final_holdout/final_holdout_predictions.csv")


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

    holdout_predictions = (
        samples.iloc[holdout_indices][["ticker", "target_date_pub", "target_beat"]]
        .copy()
        .reset_index(drop=True)
    )

    holdout_predictions["target_date_pub"] = pd.to_datetime(
        holdout_predictions["target_date_pub"]
    ).dt.strftime("%Y-%m-%d")

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

        holdout_predictions[f"{name}_prediction"] = predictions
        holdout_predictions[f"{name}_beat_probability"] = probabilities

    results["holdout_predictions"] = holdout_predictions.to_dict(orient="records")

    return results


def main() -> None:
    """runs and saves the final 2024 holdout evaluation"""

    required_files = [
        FEATURE_MATRIX_PATH,
        PREDICTION_WINDOWS_PATH,
    ]

    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(f"Required dataset not found: {file_path}")

    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    results = evaluate_final_holdout(features, samples)

    predictions = pd.DataFrame(results.pop("holdout_predictions"))

    FINAL_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FINAL_RESULTS_PATH.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    FINAL_PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        FINAL_PREDICTIONS_PATH,
        index=False,
    )

    print("Final 2024 Holdout Evaluation")
    print(f"Training samples: {results['training_samples']:,}")
    print(f"Holdout samples: {results['holdout_samples']:,}")
    print(f"Selected feature set: {results['selected_feature_set']}")

    for name, metrics in results["models"].items():
        roc_auc = metrics["roc_auc"]
        roc_auc_text = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"

        print(
            f"\n{name}: "
            f"Macro F1={metrics['macro_f1']:.4f}, "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"ROC-AUC={roc_auc_text}"
        )

    print(f"\nResults saved to: {FINAL_RESULTS_PATH}")
    print(f"Predictions saved to: {FINAL_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
