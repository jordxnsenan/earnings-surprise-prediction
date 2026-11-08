import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.benchmark_other_classifiers import (
    FEATURE_VARIANTS,
    MODEL_NAMES,
    RESULTS_DIRECTORY,
    build_model,
)
from earnings_surprise.evaluation.roc_comparison import (
    calculate_roc_results,
)
from earnings_surprise.evaluation.splits import (
    HOLDOUT_START,
    VALIDATION_YEARS,
    create_chronological_splits,
)
from earnings_surprise.features import (
    FEATURE_COLUMNS,
    FEATURE_MATRIX_PATH,
)

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
}

VARIANT_DISPLAY_NAMES = {
    "all_features": "All features",
    "mutual_information_15": "MI-selected features",
}

ROC_SUMMARY_PATH = Path("results/classification/other_classifier_roc_summary.json")

FIGURES_DIRECTORY = Path("results/figures")
AUC_TOLERANCE = 1e-6
POSITIVE_CLASS = 1
FIGURE_DPI = 200


def evaluate_classifier_roc(
    features: pd.DataFrame,
    samples: pd.DataFrame,
    model_name: str,
) -> dict:
    """calculates ROC curves across chronological validation folds"""
    if model_name not in MODEL_NAMES:
        raise ValueError("Unknown classifier")

    if (
        len(features) != len(samples)
        or not features.index.equals(samples.index)
        or features.columns.tolist() != FEATURE_COLUMNS
    ):
        raise ValueError("Feature and sample data do not match")

    if not {"target_date_pub", "target_beat"}.issubset(samples.columns):
        raise ValueError("Missing prediction sample columns")

    dates = pd.to_datetime(
        samples["target_date_pub"],
        errors="raise",
    )

    folds, holdout_positions = create_chronological_splits(samples)

    if not folds:
        raise ValueError("No chronological validation folds")

    evaluations = []

    for fold in folds:
        training_positions = np.asarray(fold["training_indices"])

        validation_positions = np.asarray(fold["validation_indices"])

        for positions in (training_positions, validation_positions):
            if (
                positions.ndim != 1
                or not np.issubdtype(positions.dtype, np.integer)
                or len(positions) == 0
                or (positions < 0).any()
                or (positions >= len(samples)).any()
                or len(np.unique(positions)) != len(positions)
            ):
                raise ValueError("Invalid chronological fold positions")

        if np.intersect1d(
            training_positions,
            validation_positions,
        ).size:
            raise ValueError("Training and validation folds overlap")

        if np.intersect1d(
            np.concatenate(
                [
                    training_positions,
                    validation_positions,
                ]
            ),
            holdout_positions,
        ).size:
            raise ValueError("Holdout observations entered validation")

        training_dates = dates.iloc[training_positions]
        validation_dates = dates.iloc[validation_positions]

        if (
            training_dates.isna().any()
            or validation_dates.isna().any()
            or (training_dates >= HOLDOUT_START).any()
            or (validation_dates >= HOLDOUT_START).any()
        ):
            raise ValueError("Holdout observations entered validation")

        if training_dates.max() >= validation_dates.min():
            raise ValueError("Training observations must precede validation")

        training_features = features.iloc[training_positions]
        validation_features = features.iloc[validation_positions]

        training_targets = samples.iloc[training_positions]["target_beat"].to_numpy()

        validation_targets = samples.iloc[validation_positions][
            "target_beat"
        ].to_numpy()

        variant_results = {}

        for variant, selected_count in FEATURE_VARIANTS.items():
            # ensures each config. receives a fresh training pipeline
            model = build_model(
                model_name,
                selected_feature_count=selected_count,
            )

            model.fit(training_features, training_targets)

            if not np.array_equal(
                np.asarray(model.classes_),
                [0, POSITIVE_CLASS],
            ):
                raise ValueError("Unexpected classifier class ordering")

            probabilities = model.predict_proba(validation_features)[:, POSITIVE_CLASS]

            variant_results[variant] = calculate_roc_results(
                validation_targets,
                probabilities,
            )

        evaluations.append(
            {
                "validation_year": int(fold["validation_year"]),
                "training_samples": len(training_positions),
                "validation_samples": len(validation_positions),
                "variants": variant_results,
            }
        )

    return {
        "model": model_name,
        "holdout_used": False,
        "holdout_samples": len(holdout_positions),
        "feature_selection_fitted_on_training_only": True,
        "folds": evaluations,
    }


def verify_and_summarize_roc(
    report: dict,
    benchmark: dict,
) -> dict:
    """verifies ROC-AUC values against saved classifier benchmarks"""
    if (
        report["model"] != benchmark["model"]
        or report["holdout_used"] is not False
        or benchmark["holdout_used"] is not False
    ):
        raise ValueError("ROC and benchmark metadata do not match")

    folds = report["folds"]

    if len(folds) != len(benchmark["by_year"]) or len(folds) == 0:
        raise ValueError("ROC validation years do not match")

    yearly_results = {}
    observed_years = []

    for fold in folds:
        year = int(fold["validation_year"])
        year_key = str(year)

        if year_key not in benchmark["by_year"]:
            raise ValueError("ROC validation years do not match")

        observed_years.append(year)
        yearly_results[year_key] = {}

        for variant in FEATURE_VARIANTS:
            actual_auc = float(fold["variants"][variant]["roc_auc"])

            expected_auc = float(benchmark["by_year"][year_key][variant]["roc_auc"])

            if not np.isclose(
                actual_auc,
                expected_auc,
                rtol=0,
                atol=AUC_TOLERANCE,
            ):
                raise ValueError("ROC-AUC differs from saved benchmark")

            yearly_results[year_key][variant] = actual_auc

    if (
        len(set(observed_years)) != len(observed_years)
        or observed_years != sorted(observed_years)
        or set(observed_years) != set(VALIDATION_YEARS)
    ):
        raise ValueError("ROC validation years do not match")

    mean_auc = {
        variant: float(
            np.mean([fold["variants"][variant]["roc_auc"] for fold in folds])
        )
        for variant in FEATURE_VARIANTS
    }

    return {
        "model": report["model"],
        "holdout_used": False,
        "holdout_samples": report["holdout_samples"],
        "feature_selection_fitted_on_training_only": True,
        "validation_years": observed_years,
        "yearly_roc_auc": yearly_results,
        "mean_roc_auc": mean_auc,
        "roc_auc_change": (
            mean_auc["mutual_information_15"] - mean_auc["all_features"]
        ),
    }


def plot_classifier_roc(
    report: dict,
    output_path: Path,
) -> None:
    """plots all-feature and MI-selected ROC curves by year"""
    folds = report["folds"]

    if not folds:
        raise ValueError("No ROC curves to plot")

    column_count = 2
    row_count = (len(folds) + column_count - 1) // column_count

    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(12, 5 * row_count),
        squeeze=False,
    )

    for ax, fold in zip(axes.flat, folds):
        for variant in FEATURE_VARIANTS:
            result = fold["variants"][variant]

            ax.plot(
                result["false_positive_rate"],
                result["true_positive_rate"],
                label=(
                    f"{VARIANT_DISPLAY_NAMES[variant]} "
                    f"(AUC={result['roc_auc']:.3f})"
                ),
            )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            label="Random classifier",
        )

        ax.set_title(f"Validation year {fold['validation_year']}")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        ax.legend()

    for ax in list(axes.flat)[len(folds) :]:
        ax.set_visible(False)

    model_label = MODEL_DISPLAY_NAMES[report["model"]]

    figure.suptitle(f"{model_label} ROC Curves — All Features vs. MI Selection")
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)


def main() -> None:
    """generates ROC figures for the remaining three classifiers"""
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    summaries = {}

    for model_name in MODEL_NAMES:
        report = evaluate_classifier_roc(
            features,
            samples,
            model_name,
        )

        benchmark_path = RESULTS_DIRECTORY / f"{model_name}_efficiency.json"

        with benchmark_path.open(encoding="utf-8") as file:
            benchmark = json.load(file)

        summary = verify_and_summarize_roc(
            report,
            benchmark,
        )

        figure_path = FIGURES_DIRECTORY / (f"{model_name}_roc_comparison.png")

        plot_classifier_roc(report, figure_path)
        summaries[model_name] = summary

        print(f"\n{MODEL_DISPLAY_NAMES[model_name]}")
        print("  Mean ROC-AUC (all): " f"{summary['mean_roc_auc']['all_features']:.4f}")
        print(
            "  Mean ROC-AUC (MI): "
            f"{summary['mean_roc_auc']['mutual_information_15']:.4f}"
        )
        print(f"  Figure saved to: {figure_path}")

    output = {
        "experiment": "other_classifier_roc_comparison",
        "holdout_used": False,
        "models": summaries,
    }

    ROC_SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ROC_SUMMARY_PATH.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nSummary saved to: {ROC_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
