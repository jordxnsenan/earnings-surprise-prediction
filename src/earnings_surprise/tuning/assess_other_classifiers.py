import argparse
import json
import numpy as np
import pandas as pd

from pathlib import Path
from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.benchmark_other_classifiers import (
    MODEL_NAMES,
    build_model,
)
from earnings_surprise.features import FEATURE_COLUMNS, FEATURE_MATRIX_PATH
from earnings_surprise.outliers.evaluate_filter_impact import (
    calculate_classification_metrics,
)
from earnings_surprise.tuning.search_other_classifiers import (
    SEARCH_PARAMETERS,
    SEARCH_SCORING,
)
from earnings_surprise.tuning.splits import (
    ASSESSMENT_YEAR,
    SEARCH_VALIDATION_YEARS,
    build_tuning_partitions,
)

RESULTS_DIRECTORY = Path("results/tuning")
TARGET_COLUMN = "target_beat"
CLASS_LABELS = (0, 1)

COMPARISON_METRICS = (
    "macro_f1",
    "accuracy",
    "roc_auc",
    "precision_class_0",
    "recall_class_0",
    "precision_class_1",
    "recall_class_1",
)


def validate_search_report(
    report: dict,
    model_name: str,
    partitions: dict,
) -> dict:
    """verifies that a saved search excluded assessment and holdout rows"""
    if (
        report.get("model") != model_name
        or report.get("scoring") != SEARCH_SCORING
        or report.get("search_validation_years") != list(SEARCH_VALIDATION_YEARS)
        or report.get("assessment_year") != ASSESSMENT_YEAR
    ):
        raise ValueError("Unexpected search configuration")

    if (
        report.get("assessment_used") is not False
        or report.get("holdout_used") is not False
        or report.get("refit_during_search") is not False
    ):
        raise ValueError("Search must exclude assessment and holdout data")

    expected_counts = {
        "search_samples": len(partitions["search_positions"]),
        "assessment_samples": len(partitions["assessment_validation_positions"]),
        "holdout_samples": len(partitions["holdout_positions"]),
    }

    for name, count in expected_counts.items():
        if report.get(name) != count:
            raise ValueError("Saved search population counts do not match")

    parameters = report["best_configuration"]["pipeline_parameters"]

    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("Missing selected hyperparameters")

    return parameters.copy()


def evaluate_model_configuration(
    features: pd.DataFrame,
    samples: pd.DataFrame,
    training_positions: np.ndarray,
    validation_positions: np.ndarray,
    model_name: str,
    parameters: dict | None = None,
) -> dict:
    """fits a classifier and evaluates predictions on the assessment rows"""
    model = build_model(
        model_name,
        selected_feature_count=None,
    )

    if parameters is not None:
        estimator_name = model.steps[-1][0]
        prefix = f"{estimator_name}__"

        available_parameters = model.get_params()
        permitted_parameters = SEARCH_PARAMETERS[model_name]

        for name, value in parameters.items():
            parameter_name = name.removeprefix(prefix)

            if (
                not name.startswith(prefix)
                or name not in available_parameters
                or parameter_name not in permitted_parameters
                or value not in permitted_parameters[parameter_name]
            ):
                raise ValueError("Unknown tuned model parameter")

        model.set_params(**parameters)

    training_features = features.iloc[training_positions]
    validation_features = features.iloc[validation_positions]

    training_targets = samples.iloc[training_positions][TARGET_COLUMN].to_numpy()

    validation_targets = samples.iloc[validation_positions][TARGET_COLUMN].to_numpy()

    model.fit(training_features, training_targets)

    if not np.array_equal(
        np.asarray(model.classes_),
        CLASS_LABELS,
    ):
        raise ValueError("Unexpected classifier class ordering")

    predictions = model.predict(validation_features)
    probabilities = model.predict_proba(validation_features)[:, 1]

    metrics = calculate_classification_metrics(
        validation_targets,
        predictions,
        probabilities,
    )

    return {
        "training_samples": len(training_positions),
        "validation_samples": len(validation_positions),
        **metrics,
    }


def assess_classifier(
    features: pd.DataFrame,
    samples: pd.DataFrame,
    search_report: dict,
    model_name: str,
) -> dict:
    """compares original and tuned classifiers on 2023 observations"""
    if model_name not in MODEL_NAMES:
        raise ValueError("Unknown classifier")

    if (
        len(features) != len(samples)
        or not features.index.equals(samples.index)
        or features.columns.tolist() != FEATURE_COLUMNS
    ):
        raise ValueError("Feature and sample data do not match")

    if not {TARGET_COLUMN, "target_date_pub"}.issubset(samples.columns):
        raise ValueError("Missing prediction sample columns")

    partitions = build_tuning_partitions(samples)

    parameters = validate_search_report(
        search_report,
        model_name,
        partitions,
    )

    training_positions = partitions["assessment_training_positions"]

    validation_positions = partitions["assessment_validation_positions"]

    holdout_positions = partitions["holdout_positions"]

    dates = pd.to_datetime(samples["target_date_pub"])

    training_dates = dates.iloc[training_positions]
    validation_dates = dates.iloc[validation_positions]

    evaluated_positions = np.concatenate(
        [
            training_positions,
            validation_positions,
        ]
    )

    if (
        training_dates.isna().any()
        or validation_dates.isna().any()
        or (training_dates.dt.year >= ASSESSMENT_YEAR).any()
        or not (validation_dates.dt.year == ASSESSMENT_YEAR).all()
        or training_dates.max() >= validation_dates.min()
        or np.intersect1d(
            evaluated_positions,
            holdout_positions,
        ).size
        > 0
    ):
        raise ValueError("Invalid assessment or holdout partition")

    baseline = evaluate_model_configuration(
        features,
        samples,
        training_positions,
        validation_positions,
        model_name,
    )

    tuned = evaluate_model_configuration(
        features,
        samples,
        training_positions,
        validation_positions,
        model_name,
        parameters=parameters,
    )

    differences = {
        metric: (
            tuned[metric] - baseline[metric]
            if (tuned[metric] is not None and baseline[metric] is not None)
            else None
        )
        for metric in COMPARISON_METRICS
    }

    return {
        "experiment": "classifier_2023_tuning_assessment",
        "model": model_name,
        "assessment_year": ASSESSMENT_YEAR,
        "training_samples": len(training_positions),
        "assessment_samples": len(validation_positions),
        "holdout_samples": len(holdout_positions),
        "holdout_used": False,
        "assessment_used_during_search": False,
        "validation_filtered": False,
        "selected_hyperparameters": parameters,
        "search_mean_macro_f1": search_report["best_configuration"]["mean_macro_f1"],
        "baseline": baseline,
        "tuned": tuned,
        "difference_tuned_minus_baseline": differences,
    }


def main() -> None:
    """assesses a saved classifier search on 2023 development data"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_NAMES,
    )

    args = parser.parse_args()

    search_path = RESULTS_DIRECTORY / f"{args.model}_random_search.json"

    with search_path.open(encoding="utf-8") as file:
        search_report = json.load(file)

    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    report = assess_classifier(
        features,
        samples,
        search_report,
        args.model,
    )

    output_path = RESULTS_DIRECTORY / f"{args.model}_2023_assessment.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"{args.model} — 2023 Tuning Assessment")
    print("------------------------------------")
    print(f"Training observations: {report['training_samples']:,}")
    print(f"Assessment observations: {report['assessment_samples']:,}")
    print(f"Reserved holdout: {report['holdout_samples']:,}")

    print("\nPerformance comparison:")

    for metric in COMPARISON_METRICS:
        baseline = report["baseline"][metric]
        tuned = report["tuned"][metric]
        difference = report["difference_tuned_minus_baseline"][metric]

        if difference is None:
            print(f"  {metric}: undefined")
            continue

        print(f"  {metric}: " f"{baseline:.5f} -> {tuned:.5f} " f"({difference:+.5f})")

    print("\nOriginal confusion matrix:")
    print(np.asarray(report["baseline"]["confusion_matrix"]))

    print("\nTuned confusion matrix:")
    print(np.asarray(report["tuned"]["confusion_matrix"]))

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
