import json
import numpy as np
import pandas as pd

from pathlib import Path
from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.features import (
    FEATURE_COLUMNS,
    FEATURE_MATRIX_PATH,
)
from earnings_surprise.models.xgboost_model import (
    build_xgboost_pipeline,
)
from earnings_surprise.outliers.evaluate_filter_impact import (
    calculate_classification_metrics,
)
from earnings_surprise.tuning.splits import (
    ASSESSMENT_YEAR,
    SEARCH_VALIDATION_YEARS,
    build_tuning_partitions,
)
from earnings_surprise.tuning.xgboost_search import (
    SEARCH_RESULTS_PATH,
    SEARCH_SCORING,
)

ASSESSMENT_RESULTS_PATH = Path("results/tuning/xgboost_2023_assessment.json")

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
    search_report: dict,
    partitions: dict,
) -> dict:
    """verifies that the saved search excludes assessment and holdout data"""
    if (
        search_report.get("model") != "xgboost_all_features"
        or search_report.get("scoring") != SEARCH_SCORING
        or search_report.get("search_validation_years") != list(SEARCH_VALIDATION_YEARS)
        or search_report.get("assessment_year") != ASSESSMENT_YEAR
    ):
        raise ValueError("Unexpected hyperparameter search configuration")

    if (
        search_report.get("assessment_used") is not False
        or search_report.get("holdout_used") is not False
        or search_report.get("refit_during_search") is not False
    ):
        raise ValueError("Search must exclude assessment and holdout data")

    expected_counts = {
        "search_samples": len(partitions["search_positions"]),
        "assessment_samples": len(partitions["assessment_validation_positions"]),
        "holdout_samples": len(partitions["holdout_positions"]),
    }

    for name, expected_count in expected_counts.items():
        if search_report.get(name) != expected_count:
            raise ValueError("Saved search population counts do not match")

    best_parameters = search_report["best_configuration"]["pipeline_parameters"]

    if not isinstance(best_parameters, dict) or not best_parameters:
        raise ValueError("Missing selected hyperparameters")

    return best_parameters.copy()


def evaluate_xgboost_variant(
    features: pd.DataFrame,
    samples: pd.DataFrame,
    training_positions: np.ndarray,
    validation_positions: np.ndarray,
    pipeline_parameters: dict | None = None,
) -> dict:
    """trains one XGBoost configuration and evaluated 2023 predictions"""
    model = build_xgboost_pipeline(selected_feature_count=None)

    if pipeline_parameters is not None:
        estimator_name = model.steps[-1][0]
        expected_prefix = f"{estimator_name}__"
        available_parameters = model.get_params()

        invalid_parameters = [
            name
            for name in pipeline_parameters
            if (
                not name.startswith(expected_prefix) or name not in available_parameters
            )
        ]

        if invalid_parameters:
            raise ValueError("Unknown tuned XGBoost parameters")

        model.set_params(**pipeline_parameters)

    training_features = features.iloc[training_positions]
    validation_features = features.iloc[validation_positions]

    training_targets = samples.iloc[training_positions][TARGET_COLUMN].to_numpy()

    validation_targets = samples.iloc[validation_positions][TARGET_COLUMN].to_numpy()

    model.fit(training_features, training_targets)

    if not np.array_equal(
        np.asarray(model.classes_),
        np.asarray(CLASS_LABELS),
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


def assess_xgboost_2023(
    features: pd.DataFrame,
    samples: pd.DataFrame,
    search_report: dict,
) -> dict:
    """compares original and tuned XGBoost models on 2023 data"""
    if (
        len(features) != len(samples)
        or not features.index.equals(samples.index)
        or features.columns.tolist() != FEATURE_COLUMNS
    ):
        raise ValueError("Feature and sample data do not match")

    if not {TARGET_COLUMN, "target_date_pub"}.issubset(samples.columns):
        raise ValueError("Missing prediction sample columns")

    partitions = build_tuning_partitions(samples)

    selected_parameters = validate_search_report(
        search_report,
        partitions,
    )

    training_positions = partitions["assessment_training_positions"]

    validation_positions = partitions["assessment_validation_positions"]

    holdout_positions = partitions["holdout_positions"]

    dates = pd.to_datetime(samples["target_date_pub"])

    training_dates = dates.iloc[training_positions]
    validation_dates = dates.iloc[validation_positions]

    if (
        training_dates.isna().any()
        or validation_dates.isna().any()
        or (training_dates.dt.year >= ASSESSMENT_YEAR).any()
        or not (validation_dates.dt.year == ASSESSMENT_YEAR).all()
        or np.intersect1d(
            np.concatenate(
                [
                    training_positions,
                    validation_positions,
                ]
            ),
            holdout_positions,
        ).size
        > 0
    ):
        raise ValueError("Invalid assessment or holdout partition")

    baseline = evaluate_xgboost_variant(
        features,
        samples,
        training_positions,
        validation_positions,
    )

    tuned = evaluate_xgboost_variant(
        features,
        samples,
        training_positions,
        validation_positions,
        pipeline_parameters=selected_parameters,
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
        "experiment": "xgboost_2023_tuning_assessment",
        "assessment_year": ASSESSMENT_YEAR,
        "training_samples": len(training_positions),
        "assessment_samples": len(validation_positions),
        "holdout_samples": len(holdout_positions),
        "holdout_used": False,
        "assessment_used_during_search": False,
        "validation_filtered": False,
        "selected_hyperparameters": selected_parameters,
        "search_mean_macro_f1": search_report["best_configuration"]["mean_macro_f1"],
        "baseline": baseline,
        "tuned": tuned,
        "difference_tuned_minus_baseline": differences,
    }


def main() -> None:
    """runs the post-search assessment and save its results"""
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    with SEARCH_RESULTS_PATH.open(encoding="utf-8") as file:
        search_report = json.load(file)

    report = assess_xgboost_2023(
        features,
        samples,
        search_report,
    )

    ASSESSMENT_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ASSESSMENT_RESULTS_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("XGBoost — 2023 Post-Search Assessment")
    print("------------------------------------")

    print(f"Training observations: {report['training_samples']:,}")
    print(f"Assessment observations: {report['assessment_samples']:,}")
    print(f"Reserved holdout: {report['holdout_samples']:,}")

    print("\nPerformance comparison:")

    for metric in COMPARISON_METRICS:
        original = report["baseline"][metric]
        tuned = report["tuned"][metric]
        difference = report["difference_tuned_minus_baseline"][metric]

        if difference is None:
            print(f"  {metric}: undefined")
            continue

        print(f"  {metric}: " f"{original:.5f} -> {tuned:.5f} " f"({difference:+.5f})")

    print("\nBaseline confusion matrix:")
    print(np.asarray(report["baseline"]["confusion_matrix"]))

    print("\nTuned confusion matrix:")
    print(np.asarray(report["tuned"]["confusion_matrix"]))

    print(f"\nResults saved to: {ASSESSMENT_RESULTS_PATH}")


if __name__ == "__main__":
    main()
