import json
import matplotlib.pyplot as plt
import numpy as np

from math import isclose, isfinite
from pathlib import Path

from earnings_surprise.tuning.splits import (
    ASSESSMENT_YEAR,
    SEARCH_VALIDATION_YEARS,
)

MODEL_NAMES = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
)

MODEL_LABELS = (
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
)

METRICS = (
    "macro_f1",
    "accuracy",
    "roc_auc",
    "precision_class_0",
    "recall_class_0",
    "precision_class_1",
    "recall_class_1",
)

RESULTS_DIRECTORY = Path("results/tuning")
SUMMARY_PATH = RESULTS_DIRECTORY / "hyperparameter_tuning_comparison.json"
FIGURE_PATH = Path("results/figures/hyperparameter_tuning_comparison.png")

SCORE_TOLERANCE = 1e-8
BAR_WIDTH = 0.35
FIGURE_DPI = 200


def summarize_tuning_model(
    name: str,
    search: dict,
    assessment: dict,
) -> dict:
    """validates saved search and assessment results for one classifier"""
    expected_search_name = "xgboost_all_features" if name == "xgboost" else name

    if (
        search.get("model") != expected_search_name
        or assessment.get("model", name) != name
        or search.get("scoring") != "f1_macro"
        or search.get("search_validation_years") != list(SEARCH_VALIDATION_YEARS)
        or search.get("assessment_year") != ASSESSMENT_YEAR
        or assessment.get("assessment_year") != ASSESSMENT_YEAR
    ):
        raise ValueError("Classifier or validation periods do not match")

    if (
        search.get("assessment_used") is not False
        or search.get("holdout_used") is not False
        or search.get("refit_during_search") is not False
        or assessment.get("holdout_used") is not False
        or assessment.get("assessment_used_during_search") is not False
        or assessment.get("validation_filtered") is not False
    ):
        raise ValueError("Search or assessment used reserved observations")

    populations = {
        "search_samples": search["search_samples"],
        "assessment_samples": assessment["assessment_samples"],
        "holdout_samples": assessment["holdout_samples"],
    }

    if (
        search["search_samples"] != assessment["training_samples"]
        or search["assessment_samples"] != assessment["assessment_samples"]
        or search["holdout_samples"] != assessment["holdout_samples"]
        or any(count <= 0 for count in populations.values())
    ):
        raise ValueError("Search and assessment populations do not match")

    selected = search["best_configuration"]
    parameters = selected["pipeline_parameters"]

    if (
        not parameters
        or parameters != assessment["selected_hyperparameters"]
        or len(search["candidates"]) != search["iterations"]
    ):
        raise ValueError("Saved search configuration does not match")

    search_baseline = float(search["baseline"]["mean_macro_f1"])
    search_tuned = float(selected["mean_macro_f1"])
    search_change = search_tuned - search_baseline

    if (
        not isfinite(search_baseline)
        or not isfinite(search_tuned)
        or not isclose(
            search_change,
            selected["improvement_over_baseline"],
            abs_tol=SCORE_TOLERANCE,
        )
        or not isclose(
            search_tuned,
            assessment["search_mean_macro_f1"],
            abs_tol=SCORE_TOLERANCE,
        )
    ):
        raise ValueError("Search scores do not match")

    baseline_metrics = {}
    tuned_metrics = {}
    differences = {}

    for metric in METRICS:
        original = float(assessment["baseline"][metric])
        tuned = float(assessment["tuned"][metric])
        difference = tuned - original

        if (
            not 0 <= original <= 1
            or not 0 <= tuned <= 1
            or not isclose(
                difference,
                assessment["difference_tuned_minus_baseline"][metric],
                abs_tol=SCORE_TOLERANCE,
            )
        ):
            raise ValueError("Assessment metrics do not match")

        baseline_metrics[metric] = original
        tuned_metrics[metric] = tuned
        differences[metric] = difference

    return {
        "populations": populations,
        "selected_hyperparameters": parameters,
        "search": {
            "candidate_count": search["iterations"],
            "baseline_macro_f1": search_baseline,
            "tuned_macro_f1": search_tuned,
            "macro_f1_change": search_change,
        },
        "assessment": {
            "baseline": baseline_metrics,
            "tuned": tuned_metrics,
            "changes": differences,
        },
    }


def build_tuning_comparison(
    search_reports: dict,
    assessment_reports: dict,
) -> dict:
    """summarizes four models evaluated on identical development periods"""
    if set(search_reports) != set(MODEL_NAMES) or set(assessment_reports) != set(
        MODEL_NAMES
    ):
        raise ValueError("Missing classifier tuning reports")

    models = {}
    shared_populations = None

    for name in MODEL_NAMES:
        summary = summarize_tuning_model(
            name,
            search_reports[name],
            assessment_reports[name],
        )

        if shared_populations is None:
            shared_populations = summary["populations"]
        elif summary["populations"] != shared_populations:
            raise ValueError("Classifier populations do not match")

        models[name] = {
            key: value for key, value in summary.items() if key != "populations"
        }

    return {
        "experiment": "four_classifier_hyperparameter_tuning_comparison",
        "search_validation_years": list(SEARCH_VALIDATION_YEARS),
        "assessment_year": ASSESSMENT_YEAR,
        "primary_metric": "macro_f1",
        "holdout_used": False,
        "populations": shared_populations,
        "models": models,
    }


def plot_tuning_comparison(report: dict, output_path: Path) -> None:
    """plots search performance and 2023 assessment trade-offs"""
    positions = np.arange(len(MODEL_NAMES))

    figure, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")

    for ax, section, title in (
        (axes[0, 0], "search", "2020–2022 search: mean macro F1"),
        (axes[0, 1], "assessment", "2023 assessment: macro F1"),
    ):
        summaries = [report["models"][name][section] for name in MODEL_NAMES]

        if section == "search":
            original = [item["baseline_macro_f1"] for item in summaries]
            tuned = [item["tuned_macro_f1"] for item in summaries]
        else:
            original = [item["baseline"]["macro_f1"] for item in summaries]
            tuned = [item["tuned"]["macro_f1"] for item in summaries]

        ax.bar(
            positions - BAR_WIDTH / 2,
            original,
            BAR_WIDTH,
            label="Original",
        )
        ax.bar(
            positions + BAR_WIDTH / 2,
            tuned,
            BAR_WIDTH,
            label="Tuned",
        )

        ax.set_title(title)
        ax.set_ylabel("Macro F1")
        ax.set_ylim(0, 0.75)
        ax.legend()

    changes = [report["models"][name]["assessment"]["changes"] for name in MODEL_NAMES]

    recall_ax = axes[1, 0]

    recall_ax.bar(
        positions - BAR_WIDTH / 2,
        [100 * result["recall_class_0"] for result in changes],
        BAR_WIDTH,
        label="Class 0",
    )
    recall_ax.bar(
        positions + BAR_WIDTH / 2,
        [100 * result["recall_class_1"] for result in changes],
        BAR_WIDTH,
        label="Class 1",
    )

    recall_ax.set_title("2023 recall changes after tuning")
    recall_ax.set_ylabel("Percentage-point change")
    recall_ax.legend()

    accuracy_ax = axes[1, 1]

    accuracy_ax.bar(
        positions,
        [100 * result["accuracy"] for result in changes],
        width=0.55,
    )

    accuracy_ax.set_title("2023 accuracy change after tuning")
    accuracy_ax.set_ylabel("Percentage-point change")

    for ax in axes.flat:
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xticks(positions, MODEL_LABELS, rotation=15)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    figure.suptitle("Hyperparameter tuning: original vs. tuned classifiers")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)


def main() -> None:
    """reads saved experiments and generates their comparison artifacts"""
    searches = {}
    assessments = {}

    for name in MODEL_NAMES:
        search_path = RESULTS_DIRECTORY / f"{name}_random_search.json"
        assessment_path = RESULTS_DIRECTORY / f"{name}_2023_assessment.json"

        with search_path.open(encoding="utf-8") as file:
            searches[name] = json.load(file)

        with assessment_path.open(encoding="utf-8") as file:
            assessments[name] = json.load(file)

    report = build_tuning_comparison(
        searches,
        assessments,
    )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    SUMMARY_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    plot_tuning_comparison(report, FIGURE_PATH)

    print("Four-Classifier Hyperparameter Tuning Comparison")
    print("------------------------------------------------")

    for name in MODEL_NAMES:
        model = report["models"][name]

        print(
            f"{name}: "
            f"search F1 {model['search']['macro_f1_change']:+.5f}, "
            f"2023 F1 {model['assessment']['changes']['macro_f1']:+.5f}, "
            f"accuracy {100 * model['assessment']['changes']['accuracy']:+.2f} pp"
        )

    print(f"\nSummary saved to: {SUMMARY_PATH}")
    print(f"Figure saved to: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
