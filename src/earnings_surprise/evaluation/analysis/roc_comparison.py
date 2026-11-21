import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from time import perf_counter
from sklearn.metrics import roc_auc_score, roc_curve

from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.splits import (
    HOLDOUT_START,
    create_chronological_splits,
)
from earnings_surprise.features import (
    FEATURE_COLUMNS,
    FEATURE_MATRIX_PATH,
)
from earnings_surprise.models.xgboost_model import (
    build_xgboost_pipeline,
)

TARGET_COLUMN = "target_beat"
POSITIVE_CLASS = 1
SELECTED_FEATURE_COUNT = 15

MODEL_VARIANTS = {
    "all_features": None,
    "mutual_information_15": SELECTED_FEATURE_COUNT,
}

ROC_RESULTS_PATH = Path("results/classification/xgboost_roc_comparison.json")

ROC_FIGURE_PATH = Path("results/figures/xgboost_roc_comparison.png")

FIGURE_DPI = 200


def calculate_roc_results(
    actual: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """calculates ROC-AUC and curve coordinates for binary predictions"""
    actual = np.asarray(actual)
    probabilities = np.asarray(probabilities, dtype=float)

    if (
        actual.ndim != 1
        or probabilities.ndim != 1
        or len(actual) == 0
        or len(actual) != len(probabilities)
    ):
        raise ValueError("ROC inputs must have matching lengths")

    if not np.isin(actual, [0, 1]).all() or len(np.unique(actual)) != 2:
        raise ValueError("ROC evaluation requires both binary classes")

    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise ValueError("Invalid prediction probabilities")

    false_positive_rate, true_positive_rate, _ = roc_curve(
        actual,
        probabilities,
        pos_label=POSITIVE_CLASS,
    )

    return {
        "roc_auc": float(roc_auc_score(actual, probabilities)),
        "false_positive_rate": (false_positive_rate.tolist()),
        "true_positive_rate": (true_positive_rate.tolist()),
    }


def evaluate_model_variant(
    training_features: pd.DataFrame,
    training_targets: np.ndarray,
    validation_features: pd.DataFrame,
    validation_targets: np.ndarray,
    selected_feature_count: int | None,
) -> dict:
    """fits one XGBoost variant and evaluate ROC and runtime"""
    fit_start = perf_counter()

    model = build_xgboost_pipeline(
        selected_feature_count=selected_feature_count,
    )

    model.fit(training_features, training_targets)

    fit_seconds = perf_counter() - fit_start

    classes = np.asarray(model.classes_)

    if not np.array_equal(classes, [0, 1]):
        raise ValueError("Unexpected classifier class ordering")

    prediction_start = perf_counter()

    probabilities = model.predict_proba(validation_features)[:, POSITIVE_CLASS]

    prediction_seconds = perf_counter() - prediction_start

    results = calculate_roc_results(
        validation_targets,
        probabilities,
    )

    results["fit_seconds"] = float(fit_seconds)
    results["predict_seconds"] = float(prediction_seconds)
    results["training_samples"] = len(training_features)
    results["validation_samples"] = len(validation_features)
    results["selected_feature_count"] = selected_feature_count

    return results


def evaluate_roc_comparison(
    features: pd.DataFrame,
    samples: pd.DataFrame,
) -> dict:
    """compares XGBoost ROC performance using chronological splits"""
    if len(features) != len(samples):
        raise ValueError("Feature and sample counts do not match")

    if not features.index.equals(samples.index):
        raise ValueError("Feature and sample indices do not match")

    if features.columns.tolist() != FEATURE_COLUMNS:
        raise ValueError("Unexpected historical feature columns")

    if TARGET_COLUMN not in samples.columns:
        raise ValueError("Missing classification target")

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

        training_dates = dates.iloc[training_positions]
        validation_dates = dates.iloc[validation_positions]

        if (
            training_dates.isna().any()
            or validation_dates.isna().any()
            or (training_dates >= HOLDOUT_START).any()
            or (validation_dates >= HOLDOUT_START).any()
            or training_dates.max() >= validation_dates.min()
            or np.intersect1d(
                training_positions,
                validation_positions,
            ).size
            > 0
        ):
            raise ValueError("Invalid chronological fold")

        training_features = features.iloc[training_positions]
        validation_features = features.iloc[validation_positions]

        training_targets = samples.iloc[training_positions][TARGET_COLUMN].to_numpy()

        validation_targets = samples.iloc[validation_positions][
            TARGET_COLUMN
        ].to_numpy()

        variant_results = {}

        for name, feature_count in MODEL_VARIANTS.items():
            variant_results[name] = evaluate_model_variant(
                training_features,
                training_targets,
                validation_features,
                validation_targets,
                selected_feature_count=feature_count,
            )

        evaluations.append(
            {
                "validation_year": int(fold["validation_year"]),
                "training_samples": len(training_positions),
                "validation_samples": len(validation_positions),
                "variants": variant_results,
            }
        )

    summary = {}

    for name in MODEL_VARIANTS:
        summary[name] = {
            "mean_roc_auc": float(
                np.mean([fold["variants"][name]["roc_auc"] for fold in evaluations])
            ),
            "mean_fit_seconds": float(
                np.mean([fold["variants"][name]["fit_seconds"] for fold in evaluations])
            ),
            "mean_predict_seconds": float(
                np.mean(
                    [fold["variants"][name]["predict_seconds"] for fold in evaluations]
                )
            ),
        }

    summary["roc_auc_difference_selected_minus_all"] = (
        summary["mutual_information_15"]["mean_roc_auc"]
        - summary["all_features"]["mean_roc_auc"]
    )

    return {
        "experiment": "xgboost_feature_selection_roc_comparison",
        "model": "xgboost",
        "holdout_used": False,
        "holdout_samples": len(holdout_positions),
        "validation_filtered": False,
        "feature_selection_fitted_on_training_only": True,
        "runtime_measurement": "wall_clock_seconds",
        "folds": evaluations,
        "summary": summary,
    }


def plot_roc_comparison(
    report: dict,
    output_path: Path,
) -> None:
    """plots both XGBoost ROC curves for each validation year"""
    folds = report["folds"]

    if not folds:
        raise ValueError("No ROC results to plot")

    column_count = 2
    row_count = (len(folds) + column_count - 1) // column_count

    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(12, 5 * row_count),
        squeeze=False,
    )

    all_axes = list(axes.flat)

    labels = {
        "all_features": "All features",
        "mutual_information_15": "MI-selected features",
    }

    for ax, fold in zip(all_axes, folds):
        for name, result in fold["variants"].items():
            ax.plot(
                result["false_positive_rate"],
                result["true_positive_rate"],
                label=(f"{labels[name]} " f"(AUC={result['roc_auc']:.3f})"),
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

    for ax in all_axes[len(folds) :]:
        ax.set_visible(False)

    figure.suptitle("XGBoost ROC Curves — All Features vs. MI Selection")
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)


def main() -> None:
    """runs the XGBoost feature-selection ROC experiment"""
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    report = evaluate_roc_comparison(features, samples)

    ROC_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ROC_RESULTS_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    plot_roc_comparison(report, ROC_FIGURE_PATH)

    print("XGBoost ROC and Feature Selection Comparison")
    print("--------------------------------------------")

    for fold in report["folds"]:
        print(f"\nValidation year: {fold['validation_year']}")

        for name, result in fold["variants"].items():
            print(
                f"  {name}: "
                f"AUC={result['roc_auc']:.4f}, "
                f"fit={result['fit_seconds']:.2f}s, "
                f"predict={result['predict_seconds']:.2f}s"
            )

    print("\nFour-year means:")

    for name in MODEL_VARIANTS:
        result = report["summary"][name]

        print(
            f"  {name}: "
            f"AUC={result['mean_roc_auc']:.4f}, "
            f"fit={result['mean_fit_seconds']:.2f}s"
        )

    difference = report["summary"]["roc_auc_difference_selected_minus_all"]

    print(f"\nROC-AUC difference: {difference:+.4f}")
    print(f"Results saved to: {ROC_RESULTS_PATH}")
    print(f"Figure saved to: {ROC_FIGURE_PATH}")


if __name__ == "__main__":
    main()
