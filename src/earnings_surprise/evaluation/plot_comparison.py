import matplotlib.pyplot as plt
import pandas as pd

from earnings_surprise.evaluation.compare_models import (
    build_comparison_tables,
    load_experiment_results,
)
from earnings_surprise.evaluation.splits import VALIDATION_YEARS
from pathlib import Path

FIGURES_DIRECTORY = Path("results/figures")
FIGURE_DPI = 250

REPRESENTATIVE_MODELS = [
    "Previous Outcome",
    "Logistic Regression - All Features",
    "Decision Tree - All Features",
    "Random Forest - MI Selection",
    "XGBoost - All Features",
]


def validate_plot_data(
    yearly: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """validates comparison data before generating figures"""
    required_yearly_columns = {
        "model",
        "validation_year",
        "macro_f1",
    }

    required_summary_columns = {
        "model",
        "mean_macro_f1",
    }

    if not required_yearly_columns.issubset(yearly.columns):
        raise ValueError("Missing yearly comparison columns")

    if not required_summary_columns.issubset(summary.columns):
        raise ValueError("Missing summary comparison columns")

    missing_models = set(REPRESENTATIVE_MODELS) - set(yearly["model"])

    if missing_models:
        raise ValueError(
            f"Missing models for yearly comparison: {sorted(missing_models)}"
        )

    if yearly.duplicated(["model", "validation_year"]).any():
        raise ValueError("Duplicate model and validation year combinations")

    actual_years = sorted(yearly["validation_year"].unique())

    if actual_years != list(VALIDATION_YEARS):
        raise ValueError("Unexpected validation years")


def plot_model_rankings(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """plots model configurations ranked by mean macro F1"""
    ranking = summary.sort_values("mean_macro_f1")

    fig, ax = plt.subplots(figsize=(10, 7))

    bars = ax.barh(
        ranking["model"],
        ranking["mean_macro_f1"],
    )

    ax.bar_label(
        bars,
        fmt="%.4f",
        padding=4,
        fontsize=9,
    )

    ax.set_xlabel("Mean Macro F1")
    ax.set_title("Classification Model Rankings (2020–2023)")

    ax.set_xlim(
        0,
        min(1.0, ranking["mean_macro_f1"].max() + 0.1),
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_yearly_performance(
    yearly: pd.DataFrame,
    output_path: Path,
) -> None:
    """compares representative models across validation years"""
    selected = yearly.loc[yearly["model"].isin(REPRESENTATIVE_MODELS)]

    comparison = selected.pivot(
        index="validation_year",
        columns="model",
        values="macro_f1",
    )

    if comparison.isna().any().any():
        raise ValueError("Missing model scores in yearly comparison")

    fig, ax = plt.subplots(figsize=(10, 6))

    years = list(VALIDATION_YEARS)

    for model_name in REPRESENTATIVE_MODELS:
        ax.plot(
            years,
            comparison.loc[years, model_name],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    lowest_score = selected["macro_f1"].min()
    highest_score = selected["macro_f1"].max()

    ax.set_ylim(
        max(0, lowest_score - 0.02),
        min(1, highest_score + 0.02),
    )

    ax.set_xticks(years)
    ax.set_xlabel("Validation Year")
    ax.set_ylabel("Macro F1 (zoomed scale)")
    ax.set_title("Model Performance Across Chronological Folds")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=9,
    )

    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


def generate_comparison_figures(
    yearly: pd.DataFrame,
    summary: pd.DataFrame,
    output_directory: Path = FIGURES_DIRECTORY,
) -> dict[str, Path]:
    """generates classification comparison figures"""
    validate_plot_data(yearly, summary)

    output_directory.mkdir(parents=True, exist_ok=True)

    ranking_path = output_directory / "model_macro_f1_ranking.png"
    yearly_path = output_directory / "model_macro_f1_by_year.png"

    plot_model_rankings(summary, ranking_path)
    plot_yearly_performance(yearly, yearly_path)

    return {
        "ranking": ranking_path,
        "yearly": yearly_path,
    }


def main() -> None:
    experiments = load_experiment_results()

    yearly, summary = build_comparison_tables(experiments)

    figures = generate_comparison_figures(yearly, summary)

    for name, path in figures.items():
        print(f"{name.capitalize()} figure saved to: {path}")


if __name__ == "__main__":
    main()
