import json
import pandas as pd

from pathlib import Path

from earnings_surprise.evaluation.splits import VALIDATION_YEARS

RESULT_PATHS = {
    "baseline": Path("results/baseline_results.json"),
    "logistic_regression": Path("results/logistic_regression_results.json"),
    "decision_tree": Path("results/decision_tree_results.json"),
    "random_forest": Path("results/random_forest_results.json"),
    "xgboost": Path("results/xgboost_results.json"),
}

EXPECTED_CONFIGURATIONS = {
    "baseline": (
        "majority_class",
        "always_beat",
        "previous_outcome",
        "historical_beat_rate",
    ),
    "logistic_regression": ("all_features", "mi_selection"),
    "decision_tree": ("all_features", "mi_selection"),
    "random_forest": ("all_features", "mi_selection"),
    "xgboost": ("all_features", "mi_selection"),
}

FAMILY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

BASELINE_NAMES = {
    "majority_class": "Majority Class",
    "always_beat": "Always Beat",
    "previous_outcome": "Previous Outcome",
    "historical_beat_rate": "Historical Beat Rate",
}

METRICS = ("accuracy", "macro_f1", "roc_auc")
RESULTS_DIRECTORY = Path("results")


def load_experiment_results(
    paths: dict[str, Path] | None = None,
) -> dict:
    """loads saved classification experiment results"""
    selected_paths = RESULT_PATHS if paths is None else paths
    experiments = {}

    for family in RESULT_PATHS:
        file_path = selected_paths[family]

        if not file_path.is_file():
            raise FileNotFoundError(f"Missing experiment results: {file_path}")

        with file_path.open(encoding="utf-8") as file:
            experiments[family] = json.load(file)

    return experiments


def validate_experiment_results(experiments: dict) -> None:
    """checks that saved experiments have compatible validation folds"""
    if set(experiments) != set(RESULT_PATHS):
        raise ValueError("Expected one results file for every model family")

    reference_folds = {}
    expected_holdout = None

    for family, results in experiments.items():
        if results.get("primary_metric") != "macro_f1":
            raise ValueError(f"Incorrect primary metric for {family}")

        holdout_samples = results["holdout_samples"]

        if expected_holdout is None:
            expected_holdout = holdout_samples
        elif holdout_samples != expected_holdout:
            raise ValueError("Holdout sample counts do not match")

        folds = results["folds"]
        years = [fold["validation_year"] for fold in folds]

        if years != list(VALIDATION_YEARS):
            raise ValueError(f"Unexpected validation years for {family}")

        for fold in folds:
            year = fold["validation_year"]
            models = fold["models"]

            if set(models) != set(EXPECTED_CONFIGURATIONS[family]):
                raise ValueError(f"Unexpected model configurations for {family}")

            training_count = fold["training_samples"]
            validation_count = fold["validation_samples"]

            for configuration, metrics in models.items():
                if metrics["samples"] != validation_count:
                    raise ValueError(
                        f"Sample count mismatch in " f"{family}/{configuration}/{year}"
                    )

                matrix = metrics["confusion_matrix"]

                if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
                    raise ValueError("Expected a 2-by-2 confusion matrix")

                meet_or_miss_count = sum(matrix[0])
                beat_count = sum(matrix[1])

                if meet_or_miss_count + beat_count != validation_count:
                    raise ValueError("Confusion matrix does not match validation count")

                supports = metrics["per_class"]

                if (
                    supports["meet_or_miss"]["support"] != meet_or_miss_count
                    or supports["beat"]["support"] != beat_count
                ):
                    raise ValueError("Class support does not match confusion matrix")

                for metric_name in METRICS:
                    value = metrics[metric_name]

                    if value is not None and not (0 <= value <= 1):
                        raise ValueError(
                            f"Invalid {metric_name} in "
                            f"{family}/{configuration}/{year}"
                        )

                fold_signature = (
                    training_count,
                    validation_count,
                    meet_or_miss_count,
                    beat_count,
                )

                if year not in reference_folds:
                    reference_folds[year] = fold_signature
                elif fold_signature != reference_folds[year]:
                    raise ValueError(
                        f"Fold coverage or class supports differ in {year}"
                    )


def get_model_name(family: str, configuration: str) -> str:
    """creates a readable name for a model configuration"""
    if family == "baseline":
        return BASELINE_NAMES[configuration]

    suffix = "All Features" if configuration == "all_features" else "MI Selection"

    return f"{FAMILY_NAMES[family]} - {suffix}"


def build_comparison_tables(
    experiments: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """craetes yearly results and model-level performance summaries"""
    validate_experiment_results(experiments)

    records = []

    for family, results in experiments.items():
        for fold in results["folds"]:
            for configuration, metrics in fold["models"].items():
                records.append(
                    {
                        "model": get_model_name(family, configuration),
                        "family": family,
                        "configuration": configuration,
                        "validation_year": fold["validation_year"],
                        "training_samples": fold["training_samples"],
                        "validation_samples": fold["validation_samples"],
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "roc_auc": metrics["roc_auc"],
                        "meet_or_miss_recall": (
                            metrics["per_class"]["meet_or_miss"]["recall"]
                        ),
                    }
                )

    yearly = pd.DataFrame.from_records(records)

    yearly = yearly.sort_values(
        ["validation_year", "model"],
        ignore_index=True,
    )

    # gives equal weight to each validation year
    summary = (
        yearly.groupby(
            ["model", "family", "configuration"],
            as_index=False,
        )
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_miss_recall=("meet_or_miss_recall", "mean"),
            std_macro_f1=("macro_f1", "std"),
            validation_years=("validation_year", "nunique"),
        )
        .sort_values(
            ["mean_macro_f1", "mean_roc_auc"],
            ascending=False,
            ignore_index=True,
        )
    )

    return yearly, summary


def main() -> None:
    """generates reproducible comparison tables from saved results"""
    experiments = load_experiment_results()
    yearly, summary = build_comparison_tables(experiments)

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    yearly.to_csv(
        RESULTS_DIRECTORY / "model_comparison_by_year.csv",
        index=False,
    )

    summary.to_csv(
        RESULTS_DIRECTORY / "model_comparison_summary.csv",
        index=False,
    )

    report = {
        "experiment": "chronological_model_comparison",
        "primary_metric": "macro_f1",
        "validation_years": list(VALIDATION_YEARS),
        "aggregation": "unweighted_mean_across_years",
        "holdout_samples": experiments["baseline"]["holdout_samples"],
        "models": summary.to_dict(orient="records"),
    }

    report_path = RESULTS_DIRECTORY / "model_comparison.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    print("Model Comparison by Mean Macro F1")
    print("---------------------------------")

    display_columns = [
        "model",
        "mean_macro_f1",
        "mean_accuracy",
        "mean_roc_auc",
    ]

    print(summary[display_columns].round(4).to_string(index=False))

    print(f"\nReserved holdout samples: " f"{report['holdout_samples']:,}")

    print(f"Results saved to: {RESULTS_DIRECTORY}")


if __name__ == "__main__":
    main()