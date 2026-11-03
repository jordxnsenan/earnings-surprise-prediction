import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from earnings_surprise.evaluation.compare_models import (
    EXPECTED_CONFIGURATIONS,
    build_comparison_tables,
    get_model_name,
    load_experiment_results,
)

ERROR_SUMMARY_PATH = Path("results/model_error_summary.csv")
FIGURES_DIRECTORY = Path("results/figures")
FIGURE_DPI = 250

COMPARISON_MODELS = (
    "Previous Outcome",
    "Logistic Regression - All Features",
    "Decision Tree - All Features",
    "Random Forest - MI Selection",
    "XGBoost - All Features",
)

CONFUSION_MODELS = (
    "Previous Outcome",
    "Random Forest - MI Selection",
    "XGBoost - All Features",
)

CLASS_NAMES = ("Meet or Miss", "Beat")


def build_error_summary(experiments: dict) -> pd.DataFrame:
    """summarizes classification errors across development folds"""
    _, comparison = build_comparison_tables(experiments)

    macro_f1_scores = comparison.set_index("model")["mean_macro_f1"]
    records = []

    for family, results in experiments.items():
        for configuration in EXPECTED_CONFIGURATIONS[family]:
            model_name = get_model_name(family, configuration)

            fold_results = [fold["models"][configuration] for fold in results["folds"]]

            confusion_matrices = [
                np.asarray(result["confusion_matrix"], dtype=int)
                for result in fold_results
            ]

            # ensures validation years don't overlap (so their counts can be pooled)
            pooled_matrix = np.sum(confusion_matrices, axis=0)

            miss_metrics = [
                result["per_class"]["meet_or_miss"] for result in fold_results
            ]

            beat_metrics = [result["per_class"]["beat"] for result in fold_results]

            records.append(
                {
                    "model": model_name,
                    "mean_macro_f1": float(macro_f1_scores[model_name]),
                    "mean_miss_precision": float(
                        np.mean([result["precision"] for result in miss_metrics])
                    ),
                    "mean_miss_recall": float(
                        np.mean([result["recall"] for result in miss_metrics])
                    ),
                    "mean_miss_f1": float(
                        np.mean([result["f1"] for result in miss_metrics])
                    ),
                    "mean_beat_recall": float(
                        np.mean([result["recall"] for result in beat_metrics])
                    ),
                    "correct_misses": int(pooled_matrix[0, 0]),
                    "misses_predicted_as_beats": int(pooled_matrix[0, 1]),
                    "beats_predicted_as_misses": int(pooled_matrix[1, 0]),
                    "correct_beats": int(pooled_matrix[1, 1]),
                    "pooled_samples": int(pooled_matrix.sum()),
                }
            )

    summary = pd.DataFrame(records)

    return summary.sort_values(
        "mean_macro_f1",
        ascending=False,
        ignore_index=True,
    )


def validate_figure_data(summary: pd.DataFrame) -> None:
    """checks that selected models have complete error statistics"""
    required_columns = {
        "model",
        "mean_miss_precision",
        "mean_miss_recall",
        "correct_misses",
        "misses_predicted_as_beats",
        "beats_predicted_as_misses",
        "correct_beats",
    }

    if not required_columns.issubset(summary.columns):
        raise ValueError("Missing error-analysis columns")

    if summary["model"].duplicated().any():
        raise ValueError("Duplicate model names in error summary")

    required_models = set(COMPARISON_MODELS) | set(CONFUSION_MODELS)
    missing_models = required_models - set(summary["model"])

    if missing_models:
        raise ValueError(f"Missing models for error analysis: {sorted(missing_models)}")


def plot_minority_class_metrics(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """compares mean meet-or-miss precision and recall"""
    selected = summary.set_index("model").loc[list(COMPARISON_MODELS)]

    positions = np.arange(len(selected))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    precision_bars = ax.barh(
        positions - bar_width / 2,
        selected["mean_miss_precision"],
        height=bar_width,
        label="Precision",
    )

    recall_bars = ax.barh(
        positions + bar_width / 2,
        selected["mean_miss_recall"],
        height=bar_width,
        label="Recall",
    )

    ax.bar_label(precision_bars, fmt="%.3f", padding=3)
    ax.bar_label(recall_bars, fmt="%.3f", padding=3)

    ax.set_yticks(positions, selected.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)

    ax.set_xlabel("Mean Score Across Validation Years")
    ax.set_title("Meet-or-Miss Precision and Recall (2020–2023)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_pooled_confusion_matrices(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """plots row-normalized confusion matrices for selected models"""
    indexed_summary = summary.set_index("model")

    fig, axes = plt.subplots(
        1,
        len(CONFUSION_MODELS),
        figsize=(13, 4.5),
        layout="constrained",
    )

    for ax, model_name in zip(axes, CONFUSION_MODELS, strict=True):
        result = indexed_summary.loc[model_name]

        counts = np.array(
            [
                [
                    result["correct_misses"],
                    result["misses_predicted_as_beats"],
                ],
                [
                    result["beats_predicted_as_misses"],
                    result["correct_beats"],
                ],
            ],
            dtype=int,
        )

        row_totals = counts.sum(axis=1, keepdims=True)

        if (row_totals == 0).any():
            raise ValueError("Both classes must have observations")

        normalized = counts / row_totals

        image = ax.imshow(
            normalized,
            cmap="Blues",
            vmin=0,
            vmax=1,
        )

        ax.set_title(model_name, fontsize=10)
        ax.set_xticks([0, 1], CLASS_NAMES)
        ax.set_yticks([0, 1], CLASS_NAMES)
        ax.set_xlabel("Predicted Outcome")

        for row_index in range(2):
            for column_index in range(2):
                proportion = normalized[row_index, column_index]
                count = counts[row_index, column_index]

                text_color = "white" if proportion >= 0.55 else "black"

                ax.text(
                    column_index,
                    row_index,
                    f"{proportion:.1%}\n({count:,})",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                )

    axes[0].set_ylabel("Actual Outcome")

    fig.suptitle(
        "Pooled Confusion Matrices (2020–2023)",
        fontsize=13,
    )

    fig.colorbar(
        image,
        ax=axes,
        shrink=0.75,
        label="Proportion of Actual Class",
    )

    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def generate_error_figures(
    summary: pd.DataFrame,
    output_directory: Path = FIGURES_DIRECTORY,
) -> dict[str, Path]:
    """generates reproducible classification-error visualizations"""
    validate_figure_data(summary)

    output_directory.mkdir(parents=True, exist_ok=True)

    precision_recall_path = output_directory / "minority_class_precision_recall.png"

    confusion_path = output_directory / "pooled_confusion_matrices.png"

    plot_minority_class_metrics(summary, precision_recall_path)
    plot_pooled_confusion_matrices(summary, confusion_path)

    return {
        "precision_recall": precision_recall_path,
        "confusion_matrices": confusion_path,
    }


def main() -> None:
    """generates error-analysis results from experiments"""
    experiments = load_experiment_results()
    summary = build_error_summary(experiments)

    ERROR_SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(ERROR_SUMMARY_PATH, index=False)

    figures = generate_error_figures(summary)

    print("Meet-or-Miss Class Performance")
    print("------------------------------")

    columns = [
        "model",
        "mean_miss_precision",
        "mean_miss_recall",
        "mean_miss_f1",
    ]

    print(summary[columns].round(4).to_string(index=False))

    print(f"\nSummary saved to: {ERROR_SUMMARY_PATH}")

    for name, path in figures.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()