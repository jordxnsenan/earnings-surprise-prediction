import json
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from earnings_surprise.evaluation.splits import VALIDATION_YEARS

RESULTS_DIRECTORY = Path("results/classification")
FIGURES_DIRECTORY = Path("results/figures")

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

MODEL_RESULT_PATHS = {
    model_name: RESULTS_DIRECTORY / f"{model_name}_efficiency.json"
    for model_name in MODEL_NAMES
}

FEATURE_VARIANTS = (
    "all_features",
    "mutual_information_15",
)

MEASUREMENTS = (
    "fit_seconds",
    "predict_seconds",
    "peak_rss_mib",
)

RESULT_PATH = RESULTS_DIRECTORY / "feature_efficiency_comparison.json"

FIGURE_PATH = FIGURES_DIRECTORY / "feature_efficiency_comparison.png"

MEASUREMENT_TOLERANCE = 1e-6
FIGURE_DPI = 200
BAR_WIDTH = 0.35


def summarize_model_benchmark(report: dict) -> dict:
    """validates repeated measurements and summarize one classifier"""
    if report.get("holdout_used") is not False:
        raise ValueError("Benchmarks must exclude the holdout")

    if report.get("memory_unit") != "MiB":
        raise ValueError("Unexpected memory measurement unit")

    repeats = report["repeats_per_fold"]

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("Invalid benchmark repeat count")

    expected_years = {str(year) for year in VALIDATION_YEARS}

    if set(report["by_year"]) != expected_years:
        raise ValueError("Validation years do not match")

    runs = report["runs"]

    expected_run_count = repeats * len(VALIDATION_YEARS) * len(FEATURE_VARIANTS)

    if len(runs) != expected_run_count:
        raise ValueError("Benchmark run count does not match")

    if any(run["holdout_used"] is not False for run in runs):
        raise ValueError("Benchmark runs must exclude the holdout")

    populations = {}
    variants = {}

    for year in VALIDATION_YEARS:
        yearly_runs = [run for run in runs if run["year"] == year]

        population_sizes = {
            (run["training_samples"], run["validation_samples"]) for run in yearly_runs
        }

        if len(population_sizes) != 1:
            raise ValueError("Validation populations do not match")

        training_count, validation_count = population_sizes.pop()

        populations[str(year)] = {
            "training_samples": int(training_count),
            "validation_samples": int(validation_count),
        }

    for variant in FEATURE_VARIANTS:
        variant_runs = [run for run in runs if run["variant"] == variant]

        yearly_auc = []

        for year in VALIDATION_YEARS:
            matching_runs = [run for run in variant_runs if run["year"] == year]

            if len(matching_runs) != repeats:
                raise ValueError("Missing benchmark repetitions")

            repeat_numbers = sorted(run["repeat"] for run in matching_runs)

            if repeat_numbers != list(range(1, repeats + 1)):
                raise ValueError("Invalid repetition numbering")

            auc_values = np.asarray([run["roc_auc"] for run in matching_runs])

            if not np.isfinite(auc_values).all() or not np.allclose(
                auc_values,
                auc_values[0],
                rtol=0,
                atol=MEASUREMENT_TOLERANCE,
            ):
                raise ValueError("Repeated ROC-AUC results changed")

            yearly_auc.append(float(auc_values.mean()))

        variant_summary = {
            "mean_roc_auc": float(np.mean(yearly_auc)),
        }

        for measurement in MEASUREMENTS:
            values = np.asarray([run[measurement] for run in variant_runs], dtype=float)

            saved_mean = report["overall"][variant][f"mean_{measurement}"]

            if (
                not np.isfinite(values).all()
                or (values < 0).any()
                or not np.isclose(
                    values.mean(),
                    saved_mean,
                    rtol=0,
                    atol=MEASUREMENT_TOLERANCE,
                )
            ):
                raise ValueError("Saved benchmark measurements changed")

            variant_summary[f"mean_{measurement}"] = float(saved_mean)

        variants[variant] = variant_summary

    all_features = variants["all_features"]
    selected = variants["mutual_information_15"]

    return {
        "repeats_per_fold": repeats,
        "populations": populations,
        "variants": variants,
        "fit_time_slowdown": (
            selected["mean_fit_seconds"] / all_features["mean_fit_seconds"]
        ),
        "peak_memory_change_mib": (
            selected["mean_peak_rss_mib"] - all_features["mean_peak_rss_mib"]
        ),
        "roc_auc_change": (selected["mean_roc_auc"] - all_features["mean_roc_auc"]),
    }


def build_feature_efficiency_comparison(
    reports: dict,
) -> dict:
    """compares four models using consistent benchmark populations"""
    if set(reports) != set(MODEL_NAMES):
        raise ValueError("Missing classifier benchmark reports")

    models = {}
    reference_populations = None
    reference_repeats = None

    for name in MODEL_NAMES:
        report = reports[name]

        if report.get("model", name) != name:
            raise ValueError("Classifier report name does not match")

        summary = summarize_model_benchmark(report)

        if reference_populations is None:
            reference_populations = summary["populations"]
            reference_repeats = summary["repeats_per_fold"]

        if (
            summary["populations"] != reference_populations
            or summary["repeats_per_fold"] != reference_repeats
        ):
            raise ValueError("Classifier benchmark settings do not match")

        models[name] = summary

    return {
        "experiment": "four_classifier_feature_efficiency",
        "validation_years": list(VALIDATION_YEARS),
        "repeats_per_fold": reference_repeats,
        "holdout_used": False,
        "feature_variants": list(FEATURE_VARIANTS),
        "fit_timing": "wall_clock_seconds_including_preprocessing",
        "memory_measurement": "whole_process_peak_rss",
        "memory_unit": "MiB",
        "models": models,
    }


def plot_feature_efficiency(
    report: dict,
    output_path: Path,
) -> None:
    """plots fitting time, peak memory ROC-AUC changes"""
    positions = np.arange(len(MODEL_NAMES))

    all_fit = []
    selected_fit = []
    all_memory = []
    selected_memory = []
    auc_changes = []

    for name in MODEL_NAMES:
        result = report["models"][name]

        all_features = result["variants"]["all_features"]
        selected = result["variants"]["mutual_information_15"]

        all_fit.append(all_features["mean_fit_seconds"])
        selected_fit.append(selected["mean_fit_seconds"])

        all_memory.append(all_features["mean_peak_rss_mib"])
        selected_memory.append(selected["mean_peak_rss_mib"])

        auc_changes.append(result["roc_auc_change"])

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
        layout="constrained",
    )

    comparisons = (
        (axes[0], all_fit, selected_fit, "Mean fitting time (s)"),
        (axes[1], all_memory, selected_memory, "Peak process memory (MiB)"),
    )

    for ax, original_values, selected_values, title in comparisons:
        ax.bar(
            positions - BAR_WIDTH / 2,
            original_values,
            width=BAR_WIDTH,
            label="All features",
        )

        ax.bar(
            positions + BAR_WIDTH / 2,
            selected_values,
            width=BAR_WIDTH,
            label="MI-selected",
        )

        ax.set_title(title)
        ax.set_xticks(positions, MODEL_LABELS, rotation=20)
        ax.grid(axis="y", alpha=0.2)
        ax.legend()

    axes[2].bar(positions, auc_changes)
    axes[2].axhline(0, color="gray", linewidth=1)
    axes[2].set_title("ROC-AUC change from MI selection")
    axes[2].set_ylabel("Selected minus all features")
    axes[2].set_xticks(positions, MODEL_LABELS, rotation=20)
    axes[2].grid(axis="y", alpha=0.2)

    figure.suptitle("Feature Selection — Predictive Performance and Efficiency")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)


def main() -> None:
    """consolidates all four saved classifier benchmarks"""
    reports = {}

    for model_name, path in MODEL_RESULT_PATHS.items():
        with path.open(encoding="utf-8") as file:
            reports[model_name] = json.load(file)

    comparison = build_feature_efficiency_comparison(reports)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )

    plot_feature_efficiency(comparison, FIGURE_PATH)

    print("Four-Classifier Feature Efficiency Comparison")
    print("---------------------------------------------")

    for name in MODEL_NAMES:
        result = comparison["models"][name]

        print(
            f"{name}: "
            f"fit slowdown={result['fit_time_slowdown']:.2f}x, "
            f"memory change={result['peak_memory_change_mib']:+.2f} MiB, "
            f"AUC change={result['roc_auc_change']:+.5f}"
        )

    print(f"\nResults saved to: {RESULT_PATH}")
    print(f"Figure saved to: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
