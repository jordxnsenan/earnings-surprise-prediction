import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.splits import (
    create_chronological_splits,
)
from earnings_surprise.features import (
    FEATURE_COLUMNS,
    FEATURE_MATRIX_PATH,
)
from earnings_surprise.models.xgboost_model import (
    build_xgboost_pipeline,
)
from earnings_surprise.outliers.training_filter import (
    filter_training_fold,
)

TARGET_COLUMN = "target_beat"

MEET_OR_MISS_CLASS = 0
BEAT_CLASS = 1
CLASS_LABELS = (MEET_OR_MISS_CLASS, BEAT_CLASS)

EVALUATION_METRICS = (
    "accuracy",
    "macro_f1",
    "roc_auc",
    "precision_class_0",
    "recall_class_0",
    "precision_class_1",
    "recall_class_1",
)

FILTER_IMPACT_PATH = Path("results/outliers/xgboost_filter_impact.json")


def calculate_classification_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """calculates binary classification metrics for one validation fold"""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    probabilities = np.asarray(probabilities, dtype=float)

    if (
        actual.ndim != 1
        or predicted.ndim != 1
        or probabilities.ndim != 1
        or len(actual) == 0
        or len(actual) != len(predicted)
        or len(actual) != len(probabilities)
    ):
        raise ValueError("Classification outputs must have matching lengths")

    if (
        not np.isin(actual, CLASS_LABELS).all()
        or not np.isin(predicted, CLASS_LABELS).all()
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise ValueError("Invalid classification outputs")

    precision, recall, _, _ = precision_recall_fscore_support(
        actual,
        predicted,
        labels=CLASS_LABELS,
        zero_division=0,
    )

    has_both_classes = len(np.unique(actual)) == len(CLASS_LABELS)

    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                average="macro",
                labels=CLASS_LABELS,
                zero_division=0,
            )
        ),
        "roc_auc": (
            float(roc_auc_score(actual, probabilities)) if has_both_classes else None
        ),
        "precision_class_0": float(precision[0]),
        "recall_class_0": float(recall[0]),
        "precision_class_1": float(precision[1]),
        "recall_class_1": float(recall[1]),
        "confusion_matrix": confusion_matrix(
            actual,
            predicted,
            labels=CLASS_LABELS,
        ).tolist(),
    }


def evaluate_xgboost_fold(
    features: pd.DataFrame,
    samples: pd.DataFrame,
    training_positions: np.ndarray,
    validation_positions: np.ndarray,
) -> dict:
    """trains fresh XGBoost model and evaluate its validation predictions"""
    training_features = features.iloc[training_positions]
    validation_features = features.iloc[validation_positions]

    training_targets = samples.iloc[training_positions][TARGET_COLUMN].to_numpy()

    validation_targets = samples.iloc[validation_positions][TARGET_COLUMN].to_numpy()

    if len(np.unique(training_targets)) != len(CLASS_LABELS):
        raise ValueError("Training data must contain both target classes")

    model = build_xgboost_pipeline(selected_feature_count=None)

    model.fit(training_features, training_targets)

    if not np.array_equal(
        np.asarray(model.classes_),
        np.asarray(CLASS_LABELS),
    ):
        raise ValueError("Unexpected classifier class ordering")

    predictions = model.predict(validation_features)

    probabilities = model.predict_proba(validation_features)[:, BEAT_CLASS]

    return calculate_classification_metrics(
        validation_targets,
        predictions,
        probabilities,
    )


def summarize_fold_performance(
    folds: list[dict],
    condition: str,
) -> dict:
    """averages the classification metrics across validation years"""
    averages = {}

    for metric in EVALUATION_METRICS:
        values = [
            fold[condition][metric]
            for fold in folds
            if fold[condition][metric] is not None
        ]

        averages[metric] = float(np.mean(values)) if values else None

    return averages


def evaluate_filter_impact(
    features: pd.DataFrame,
    samples: pd.DataFrame,
) -> dict:
    """compares full and outlier-filtered training across chronological folds"""
    if len(features) != len(samples):
        raise ValueError("Feature and sample counts do not match")

    if not features.index.equals(samples.index):
        raise ValueError("Feature and sample indices do not match")

    if features.columns.tolist() != FEATURE_COLUMNS:
        raise ValueError("Unexpected historical feature columns")

    if TARGET_COLUMN not in samples.columns:
        raise ValueError("Missing classification target")

    chronological_folds, holdout_indices = create_chronological_splits(samples)

    if not chronological_folds:
        raise ValueError("No chronological validation folds")

    fold_results = []

    for fold in chronological_folds:
        training_positions = np.asarray(fold["training_indices"])

        validation_positions = np.asarray(fold["validation_indices"])

        # fits preprocessing and Isolation Forest on training rows
        retained_positions, filter_summary = filter_training_fold(
            features,
            samples,
            training_positions,
            validation_positions,
        )

        # ensures fresh models keep both experimental conditions independent
        baseline_metrics = evaluate_xgboost_fold(
            features,
            samples,
            training_positions,
            validation_positions,
        )

        filtered_metrics = evaluate_xgboost_fold(
            features,
            samples,
            retained_positions,
            validation_positions,
        )

        fold_results.append(
            {
                "validation_year": int(fold["validation_year"]),
                "training_samples": len(training_positions),
                "validation_samples": len(validation_positions),
                "retained_training_samples": len(retained_positions),
                "removed_training_samples": filter_summary["removed_training_samples"],
                "removed_fraction": filter_summary["removed_fraction"],
                "filter_summary": filter_summary,
                "baseline": baseline_metrics,
                "filtered": filtered_metrics,
                "macro_f1_difference": (
                    filtered_metrics["macro_f1"] - baseline_metrics["macro_f1"]
                ),
            }
        )

    baseline_means = summarize_fold_performance(
        fold_results,
        "baseline",
    )

    filtered_means = summarize_fold_performance(
        fold_results,
        "filtered",
    )

    differences = {
        metric: (
            filtered_means[metric] - baseline_means[metric]
            if (
                filtered_means[metric] is not None
                and baseline_means[metric] is not None
            )
            else None
        )
        for metric in EVALUATION_METRICS
    }

    return {
        "experiment": "xgboost_outlier_removal_impact",
        "model": "xgboost_all_features",
        "outlier_detector": "isolation_forest",
        "primary_metric": "macro_f1",
        "validation_years": [fold["validation_year"] for fold in fold_results],
        "holdout_samples": len(holdout_indices),
        "holdout_used": False,
        "validation_filtered": False,
        "folds": fold_results,
        "mean_performance": {
            "baseline": baseline_means,
            "filtered": filtered_means,
            "difference_filtered_minus_baseline": differences,
        },
    }


def main() -> None:
    """runs the chronological XGBoost outlier-removal experiment"""
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    results = evaluate_filter_impact(
        features,
        samples,
    )

    FILTER_IMPACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FILTER_IMPACT_PATH.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print("XGBoost — Impact of Training-Only Outlier Removal")
    print("------------------------------------------------")

    for fold in results["folds"]:
        print(f"\nValidation year: {fold['validation_year']}")
        print(f"Training samples: {fold['training_samples']:,}")
        print(
            "Removed training samples: "
            f"{fold['removed_training_samples']:,} "
            f"({fold['removed_fraction']:.2%})"
        )
        print(f"Validation samples: {fold['validation_samples']:,}")

        print("Baseline macro F1: " f"{fold['baseline']['macro_f1']:.4f}")
        print("Filtered macro F1: " f"{fold['filtered']['macro_f1']:.4f}")
        print("Macro F1 difference: " f"{fold['macro_f1_difference']:+.4f}")

    means = results["mean_performance"]

    print("\nMean Performance Across Validation Years")
    print("----------------------------------------")

    for metric in EVALUATION_METRICS:
        baseline = means["baseline"][metric]
        filtered = means["filtered"][metric]
        difference = means["difference_filtered_minus_baseline"][metric]

        if baseline is None:
            print(f"{metric}: undefined")
            continue

        print(f"{metric}: " f"{baseline:.4f} -> {filtered:.4f} " f"({difference:+.4f})")

    print(f"\nResults saved to: {FILTER_IMPACT_PATH}")


if __name__ == "__main__":
    main()
