import json
from pathlib import Path

import pandas as pd

from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.metrics import evaluate_predictions
from earnings_surprise.evaluation.splits import (
    create_chronological_splits,
)
from earnings_surprise.features import FEATURE_MATRIX_PATH
from earnings_surprise.models.baselines import (
    AlwaysBeatBaseline,
    HistoricalBeatRateBaseline,
    MajorityClassBaseline,
    PreviousOutcomeBaseline,
)


BASELINE_RESULTS_PATH = Path("results/baseline_results.json")

REQUIRED_FEATURES = ["beat_lag1", "beat_rate4"]

BASELINE_MODELS = {
    "majority_class": MajorityClassBaseline,
    "always_beat": AlwaysBeatBaseline,
    "previous_outcome": PreviousOutcomeBaseline,
    "historical_beat_rate": HistoricalBeatRateBaseline,
}


def evaluate_baselines(
    features: pd.DataFrame,
    samples: pd.DataFrame,
) -> dict:
    """evaluates baseline classifiers across chronological folds"""
    if len(features) != len(samples):
        raise ValueError("Feature and sample counts do not match")

    if not features.index.equals(samples.index):
        raise ValueError("Feature and sample indices do not match")

    missing_columns = [
        column for column in REQUIRED_FEATURES
        if column not in features.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required features: {missing_columns}"
        )

    folds, holdout_indices = create_chronological_splits(samples)

    results = {
        "experiment": "chronological_baseline_evaluation",
        "primary_metric": "macro_f1",
        "holdout_samples": len(holdout_indices),
        "folds": [],
    }

    for fold in folds:
        year = fold["validation_year"]

        training_indices = fold["training_indices"]
        validation_indices = fold["validation_indices"]

        X_train = features.iloc[training_indices]
        X_validation = features.iloc[validation_indices]

        y_train = samples.iloc[training_indices]["target_beat"]
        y_validation = samples.iloc[validation_indices]["target_beat"]

        fold_results = {
            "validation_year": year,
            "training_samples": len(training_indices),
            "validation_samples": len(validation_indices),
            "models": {},
        }

        for name, model_class in BASELINE_MODELS.items():
            
            # ensures each fold receives a fresh model instance
            model = model_class()
            model.fit(X_train, y_train)

            predictions = model.predict(X_validation)
            beat_scores = model.predict_proba(X_validation)[:, 1]

            metrics = evaluate_predictions(
                y_validation,
                predictions,
                beat_scores,
            )

            fold_results["models"][name] = metrics

        results["folds"].append(fold_results)

    return results


def main() -> None:
    """runs and saves all chronological baseline experiments"""
    required_files = [
        FEATURE_MATRIX_PATH,
        PREDICTION_WINDOWS_PATH,
    ]

    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required dataset not found: {file_path}"
            )

    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    results = evaluate_baselines(features, samples)

    BASELINE_RESULTS_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    BASELINE_RESULTS_PATH.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    for fold in results["folds"]:
        year = fold["validation_year"]

        print(f"\nValidation Year: {year}")
        print(f"Validation samples: {fold['validation_samples']:,}")

        for name, metrics in fold["models"].items():
            print(
                f"  {name}: "
                f"Accuracy={metrics['accuracy']:.4f}, "
                f"Macro F1={metrics['macro_f1']:.4f}, "
                f"ROC-AUC={metrics['roc_auc']:.4f}"
            )

    print(f"\nReserved holdout: {results['holdout_samples']:,}")
    print(f"Results saved to: {BASELINE_RESULTS_PATH}")


if __name__ == "__main__":
    main()