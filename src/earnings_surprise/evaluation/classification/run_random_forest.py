import json
from pathlib import Path

import numpy as np
import pandas as pd

from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.metrics import evaluate_predictions
from earnings_surprise.evaluation.splits import create_chronological_splits
from earnings_surprise.features import FEATURE_COLUMNS, FEATURE_MATRIX_PATH
from earnings_surprise.models.random_forest import (
    FOREST_MAX_DEPTH,
    FOREST_MAX_FEATURES,
    FOREST_MIN_SAMPLES_LEAF,
    FOREST_RANDOM_STATE,
    FOREST_TREE_COUNT,
    build_random_forest_pipeline,
)
from earnings_surprise.selection import get_selected_feature_names


FOREST_RESULTS_PATH = Path("results/random_forest_results.json")

SELECTED_FEATURE_COUNT = 15

MODEL_CONFIGURATIONS = {
    "all_features": None,
    "mi_selection": SELECTED_FEATURE_COUNT,
}

EVALUATION_METRICS = ("accuracy", "macro_f1", "roc_auc")


def evaluate_random_forest_models(
    features: pd.DataFrame,
    samples: pd.DataFrame,
) -> dict:
    """evaluates Random Forest configurations across chronological folds"""
    if len(features) != len(samples):
        raise ValueError("Feature and sample counts do not match")

    if not features.index.equals(samples.index):
        raise ValueError("Feature and sample indices do not match")

    if features.columns.tolist() != FEATURE_COLUMNS:
        raise ValueError("Feature columns do not match the specification")

    folds, holdout_indices = create_chronological_splits(samples)

    results = {
        "experiment": "chronological_random_forest",
        "primary_metric": "macro_f1",
        "model_configuration": {
            "tree_count": FOREST_TREE_COUNT,
            "maximum_depth": FOREST_MAX_DEPTH,
            "minimum_samples_per_leaf": FOREST_MIN_SAMPLES_LEAF,
            "max_features": FOREST_MAX_FEATURES,
            "random_state": FOREST_RANDOM_STATE,
        },
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
            model = build_random_forest_pipeline(
                selected_feature_count=selected_count
            )

            model.fit(X_train, y_train)

            classifier = model.named_steps["classifier"]
            preprocessor = model.named_steps["preprocessor"]

            if not np.array_equal(classifier.classes_, [0, 1]):
                raise ValueError(
                    f"Training fold {year} must contain both classes"
                )

            predictions = model.predict(X_validation)
            probabilities = model.predict_proba(X_validation)[:, 1]

            if not np.isfinite(probabilities).all():
                raise ValueError("Non-finite prediction probabilities")

            metrics = evaluate_predictions(
                y_validation,
                predictions,
                probabilities,
            )

            transformed_names = preprocessor.get_feature_names_out()
            selected_features = []

            if selected_count is None:
                feature_names = list(transformed_names)
            else:
                selected_features = get_selected_feature_names(
                    transformed_names,
                    model.named_steps["selector"],
                )
                feature_names = selected_features

            if len(feature_names) != classifier.n_features_in_:
                raise ValueError("Forest feature names are misaligned")

            feature_importances = [
                {
                    "feature": str(feature_name),
                    "importance": float(importance),
                }
                for feature_name, importance in zip(
                    feature_names,
                    classifier.feature_importances_,
                    strict=True,
                )
            ]

            feature_importances.sort(
                key=lambda item: item["importance"],
                reverse=True,
            )

            tree_depths = [
                tree.get_depth()
                for tree in classifier.estimators_
            ]

            fold_results["models"][name] = {
                **metrics,
                "feature_count": int(classifier.n_features_in_),
                "tree_count": len(classifier.estimators_),
                "mean_tree_depth": float(np.mean(tree_depths)),
                "maximum_tree_depth": int(max(tree_depths)),
                "selected_features": selected_features,
                "feature_importances": feature_importances,
            }

        results["folds"].append(fold_results)

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
    """runs and saves the chronological Random Forest experiment"""
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

    results = evaluate_random_forest_models(features, samples)

    FOREST_RESULTS_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    FOREST_RESULTS_PATH.write_text(
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
                f"ROC-AUC={metrics['roc_auc']:.4f}, "
                f"Mean Depth={metrics['mean_tree_depth']:.2f}"
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
    print(f"Results saved to: {FOREST_RESULTS_PATH}")


if __name__ == "__main__":
    main()