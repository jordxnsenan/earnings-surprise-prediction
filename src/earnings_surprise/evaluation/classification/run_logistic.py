import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning

from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.metrics import evaluate_predictions
from earnings_surprise.evaluation.splits import (
    create_chronological_splits,
)
from earnings_surprise.features import (
    FEATURE_COLUMNS,
    FEATURE_MATRIX_PATH,
)
from earnings_surprise.models.logistic import (
    LOGISTIC_MAX_ITERATIONS,
    LOGISTIC_REGULARIZATION_STRENGTH,
    build_logistic_regression_pipeline,
)
from earnings_surprise.selection import get_selected_feature_names


LOGISTIC_RESULTS_PATH = Path("results/logistic_regression_results.json")

SELECTED_FEATURE_COUNT = 15

MODEL_CONFIGURATIONS = {
    "all_features": None,
    "mi_selection": SELECTED_FEATURE_COUNT,
}

EVALUATION_METRICS = ["accuracy", "macro_f1", "roc_auc"]


def evaluate_logistic_models(
    features: pd.DataFrame,
    samples: pd.DataFrame,
) -> dict:
    """evaluates Logistic Regression across chronological folds"""
    if len(features) != len(samples):
        raise ValueError("Feature and sample counts do not match")

    if not features.index.equals(samples.index):
        raise ValueError("Feature and sample indices do not match")

    if features.columns.tolist() != FEATURE_COLUMNS:
        raise ValueError("Feature columns do not match the specification")

    folds, holdout_indices = create_chronological_splits(samples)

    results = {
        "experiment": "chronological_logistic_regression",
        "primary_metric": "macro_f1",
        "continuous_scaler": "quantile",
        "regularization_C": LOGISTIC_REGULARIZATION_STRENGTH,
        "maximum_iterations": LOGISTIC_MAX_ITERATIONS,
        "holdout_samples": len(holdout_indices),
        "folds": [],
    }

    for fold in folds:
        year = fold["validation_year"]

        training_indices = fold["training_indices"]
        validation_indices = fold["validation_indices"]

        X_train = features.iloc[training_indices]
        X_validation = features.iloc[validation_indices]

        y_train = samples.iloc[
            training_indices
        ]["target_beat"].astype(int)

        y_validation = samples.iloc[
            validation_indices
        ]["target_beat"].astype(int)

        fold_results = {
            "validation_year": year,
            "training_samples": len(training_indices),
            "validation_samples": len(validation_indices),
            "models": {},
        }

        for name, selected_count in MODEL_CONFIGURATIONS.items():
            # ensures each config. and fold receives fresh pipeline
            model = build_logistic_regression_pipeline(
                continuous_scaler="quantile",
                selected_feature_count=selected_count,
            )

            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(X_train, y_train)

            convergence_warnings = [
                warning
                for warning in recorded
                if issubclass(warning.category, ConvergenceWarning)
            ]

            if convergence_warnings:
                raise RuntimeError(
                    f"Logistic Regression did not converge: "
                    f"{name}, validation year {year}"
                )

            classifier = model.named_steps["classifier"]

            if not np.isfinite(classifier.coef_).all():
                raise ValueError("Non-finite model coefficients")

            predictions = model.predict(X_validation)
            probabilities = model.predict_proba(X_validation)[:, 1]

            metrics = evaluate_predictions(
                y_validation,
                predictions,
                probabilities,
            )

            selected_features = []

            if selected_count is not None:
                transformed_names = (
                    model.named_steps["preprocessor"]
                    .get_feature_names_out()
                )

                selected_features = get_selected_feature_names(
                    transformed_names,
                    model.named_steps["selector"],
                )

            fold_results["models"][name] = {
                **metrics,
                "feature_count": int(classifier.coef_.shape[1]),
                "iterations": int(classifier.n_iter_[0]),
                "convergence_warnings": len(convergence_warnings),
                "selected_features": selected_features,
            }

        results["folds"].append(fold_results)

    # calculates unweighted mean across validation years
    mean_performance = {}

    for name in MODEL_CONFIGURATIONS:
        mean_performance[name] = {}

        for metric in EVALUATION_METRICS:
            scores = [
                fold["models"][name][metric]
                for fold in results["folds"]
                if fold["models"][name][metric] is not None
            ]

            mean_performance[name][metric] = (
                float(np.mean(scores)) if scores else None
            )

    results["mean_performance"] = mean_performance

    return results


def main() -> None:
    """runs and saves the chronological Logistic Regression experiment"""
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

    results = evaluate_logistic_models(features, samples)

    LOGISTIC_RESULTS_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    LOGISTIC_RESULTS_PATH.write_text(
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
    print(f"Results saved to: {LOGISTIC_RESULTS_PATH}")


if __name__ == "__main__":
    main()