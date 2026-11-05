import json
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from earnings_surprise.outliers.evaluate_filter_impact import (
    FILTER_IMPACT_PATH,
)

FILTER_IMPACT_FIGURE_PATH = Path("results/figures/xgboost_outlier_filter_impact.png")

FIGURE_DPI = 200
BAR_WIDTH = 0.35
METRIC_TOLERANCE = 1e-9


def summarize_filter_impact(report: dict) -> dict:
    """validates and summarizes the chronological filtering experiment"""
    if (
        report.get("holdout_used") is not False
        or report.get("validation_filtered") is not False
    ):
        raise ValueError("Holdout and validation data must remain unchanged")

    folds = report.get("folds", [])

    if not folds:
        raise ValueError("No validation folds available")

    years = []
    baseline_f1 = []
    filtered_f1 = []
    class_0_recall_changes = []

    for fold in folds:
        if (
            fold["filter_summary"]["holdout_used"] is not False
            or fold["filter_summary"]["validation_filtered"] is not False
        ):
            raise ValueError("Filtering must affect training rows only")

        if (
            fold["retained_training_samples"] + fold["removed_training_samples"]
            != fold["training_samples"]
        ):
            raise ValueError("Training sample counts do not match")

        baseline = fold["baseline"]
        filtered = fold["filtered"]

        values = (
            baseline["macro_f1"],
            filtered["macro_f1"],
            baseline["recall_class_0"],
            filtered["recall_class_0"],
        )

        if not all(np.isfinite(value) and 0 <= value <= 1 for value in values):
            raise ValueError("Invalid classification metrics")

        difference = filtered["macro_f1"] - baseline["macro_f1"]

        if not np.isclose(
            difference,
            fold["macro_f1_difference"],
            rtol=0,
            atol=METRIC_TOLERANCE,
        ):
            raise ValueError("Macro F1 difference does not match")

        years.append(int(fold["validation_year"]))
        baseline_f1.append(float(baseline["macro_f1"]))
        filtered_f1.append(float(filtered["macro_f1"]))

        class_0_recall_changes.append(
            float(filtered["recall_class_0"] - baseline["recall_class_0"])
        )

    if len(set(years)) != len(years) or years != sorted(years):
        raise ValueError("Validation years must be unique and ordered")

    f1_changes = np.subtract(filtered_f1, baseline_f1)

    mean_difference = float(np.mean(f1_changes))

    saved_difference = report["mean_performance"]["difference_filtered_minus_baseline"][
        "macro_f1"
    ]

    if not np.isclose(
        mean_difference,
        saved_difference,
        rtol=0,
        atol=METRIC_TOLERANCE,
    ):
        raise ValueError("Saved mean macro F1 difference changed")

    return {
        "years": years,
        "baseline_macro_f1": baseline_f1,
        "filtered_macro_f1": filtered_f1,
        "macro_f1_changes": f1_changes.tolist(),
        "class_0_recall_changes": class_0_recall_changes,
        "mean_macro_f1_difference": mean_difference,
    }


def plot_filter_impact(
    report: dict,
    output_path: Path,
) -> dict:
    """visualizes yearly classification performance before and after filtering"""
    summary = summarize_filter_impact(report)

    years = [str(year) for year in summary["years"]]
    positions = np.arange(len(years))

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        layout="constrained",
    )

    axes[0].plot(
        positions,
        summary["baseline_macro_f1"],
        marker="o",
        label="All training observations",
    )

    axes[0].plot(
        positions,
        summary["filtered_macro_f1"],
        marker="s",
        label="Outliers removed",
    )

    axes[0].set_title("XGBoost macro F1 by validation year")
    axes[0].set_ylabel("Macro F1")
    axes[0].set_xticks(positions, years)
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].bar(
        positions - BAR_WIDTH / 2,
        np.asarray(summary["macro_f1_changes"]) * 100,
        width=BAR_WIDTH,
        label="Macro F1 change",
    )

    axes[1].bar(
        positions + BAR_WIDTH / 2,
        np.asarray(summary["class_0_recall_changes"]) * 100,
        width=BAR_WIDTH,
        label="Class 0 recall change",
    )

    axes[1].axhline(0, linewidth=1)
    axes[1].set_title("Impact of removing training outliers")
    axes[1].set_ylabel("Change (percentage points)")
    axes[1].set_xticks(positions, years)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)

    return summary


def main() -> None:
    """generates the final outlier-removal impact figure"""
    with FILTER_IMPACT_PATH.open(encoding="utf-8") as file:
        report = json.load(file)

    summary = plot_filter_impact(
        report,
        FILTER_IMPACT_FIGURE_PATH,
    )

    print("Outlier Removal — Classification Impact")
    print("---------------------------------------")
    print(f"Validation years: " f"{', '.join(map(str, summary['years']))}")
    print(f"Mean macro F1 change: " f"{summary['mean_macro_f1_difference']:+.6f}")
    print(f"Figure saved to: {FILTER_IMPACT_FIGURE_PATH}")


if __name__ == "__main__":
    main()
