import json
import numpy as np
import pandas as pd

from pathlib import Path
from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.ablation_features import FEATURE_SETS
from earnings_surprise.evaluation.metrics import evaluate_predictions
from earnings_surprise.evaluation.splits import create_chronological_splits
from earnings_surprise.features import FEATURE_COLUMNS, FEATURE_MATRIX_PATH
from earnings_surprise.models.xgboost_model import build_xgboost_pipeline

ABLATION_RESULTS_PATH = Path("results/ablations/xgboost_feature_ablations.json")

EVALUATION_METRICS = ("accuracy", "macro_f1", "roc_auc")


def evaluate_feature_ablations(
    features: pd.DataFrame,
    samples: pd.DataFrame,
) -> dict:
    """evaluates XGBoost using different historical feature sets"""
    if len(features) != len(samples):
        raise ValueError("Feature and sample counts do not match")

    if not features.index.equals(samples.index):
        raise ValueError("Feature and sample indices do not match")

    if features.columns.tolist() != FEATURE_COLUMNS:
        raise ValueError("Feature columns do not match the specification")

    folds, holdout_indices = create_chronological_splits(samples)

    results = {
        "experiment": "chronological_xgboost_feature_ablation",
        "primary_metric": "macro_f1",
        "holdout_samples": len(holdout_indices),
        "holdout_used": False,
        "feature_sets": FEATURE_SETS,
        "folds": [],
    }

    for fold in folds:
        year = fold["validation_year"]

        training_indices = fold["training_indices"]
        validation_indices = fold["validation_indices"]

        y_train = samples.iloc[training_indices]["target_beat"].astype(int)
        y_validation = samples.iloc[validation_indices]["target_beat"].astype(int)

        fold_results = {
            "validation_year": year,
            "training_samples": len(training_indices),
            "validation_samples": len(validation_indices),
            "models": {},
        }

        for name, feature_columns in FEATURE_SETS.items():
            X_train = features.iloc[training_indices][feature_columns]
            X_validation = features.iloc[validation_indices][feature_columns]

            model = build_xgboost_pipeline(feature_columns=feature_columns)

            model.fit(X_train, y_train)

            predictions = model.predict(X_validation)
            probabilities = model.predict_proba(X_validation)[:, 1]

            if not np.isfinite(probabilities).all():
                raise ValueError("Non-finite prediction probabilities")

            metrics = evaluate_predictions(
                y_validation,
                predictions,
                probabilities,
            )

            classifier = model.named_steps["classifier"]

            fold_results["models"][name] = {
                **metrics,
                "original_feature_count": len(feature_columns),
                "transformed_feature_count": int(classifier.n_features_in_),
            }

        results["folds"].append(fold_results)

    mean_performance = {}

    for name in FEATURE_SETS:
        mean_performance[name] = {}

        for metric in EVALUATION_METRICS:
            scores = [fold["models"][name][metric] for fold in results["folds"]]

            mean_performance[name][metric] = float(np.mean(scores))

    results["mean_performance"] = mean_performance

    return results


def main() -> None:
    """runs and saves the historical feature ablation experiment"""
    required_files = [
        FEATURE_MATRIX_PATH,
        PREDICTION_WINDOWS_PATH,
    ]

    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(f"Required dataset not found: {file_path}")

    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    results = evaluate_feature_ablations(features, samples)

    ABLATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    ABLATION_RESULTS_PATH.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    for fold in results["folds"]:
        print(f"\nValidation Year: {fold['validation_year']}")

        for name, metrics in fold["models"].items():
            print(
                f"  {name}: "
                f"Macro F1={metrics['macro_f1']:.4f}, "
                f"Accuracy={metrics['accuracy']:.4f}, "
                f"ROC-AUC={metrics['roc_auc']:.4f}"
            )

    print("\nMean Performance")

    for name, metrics in results["mean_performance"].items():
        print(
            f"  {name}: "
            f"Macro F1={metrics['macro_f1']:.4f}, "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}"
        )

    print(f"\nReserved holdout samples: {results['holdout_samples']:,}")
    print(f"Results saved to: {ABLATION_RESULTS_PATH}")


if __name__ == "__main__":
    main()
