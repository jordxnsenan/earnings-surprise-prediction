import argparse
import json
import numpy as np
import pandas as pd
import platform
import resource
import subprocess
import sys

from pathlib import Path
from sklearn.metrics import roc_auc_score
from statistics import mean, stdev
from time import perf_counter

from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.splits import (
    HOLDOUT_START,
    VALIDATION_YEARS,
    create_chronological_splits,
)
from earnings_surprise.features import (
    FEATURE_COLUMNS,
    FEATURE_MATRIX_PATH,
)
from earnings_surprise.models.logistic import (
    build_logistic_regression_pipeline,
)
from earnings_surprise.models.decision_tree import (
    build_decision_tree_pipeline,
)
from earnings_surprise.models.random_forest import (
    build_random_forest_pipeline,
)

MODEL_NAMES = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
)

FEATURE_VARIANTS = {
    "all_features": None,
    "mutual_information_15": 15,
}

BENCHMARK_REPEATS = 3
WORKER_RESULT_PREFIX = "BENCHMARK_RESULT:"

BENCHMARK_MEASUREMENTS = (
    "fit_seconds",
    "predict_seconds",
    "peak_rss_mib",
)

RESULTS_DIRECTORY = Path("results/classification")
AUC_TOLERANCE = 1e-6


def build_model(
    model_name: str,
    selected_feature_count: int | None,
):
    """builds an existing classifier with consistent settings"""
    if model_name == "logistic_regression":
        return build_logistic_regression_pipeline(
            continuous_scaler="quantile",
            selected_feature_count=selected_feature_count,
        )

    if model_name == "decision_tree":
        return build_decision_tree_pipeline(
            selected_feature_count=selected_feature_count,
        )

    if model_name == "random_forest":
        return build_random_forest_pipeline(
            selected_feature_count=selected_feature_count,
        )

    raise ValueError("Unknown classifier")


def get_peak_memory_mib() -> float:
    """measures peak resident memory of the current process"""
    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if platform.system() == "Darwin":
        return float(peak_memory / (1024 * 1024))

    return float(peak_memory / 1024)


def benchmark_one_fit(
    model_name: str,
    year: int,
    variant: str,
) -> dict:
    """benchmarks one classifier in a fresh process"""
    if model_name not in MODEL_NAMES:
        raise ValueError("Unknown classifier")

    if year not in VALIDATION_YEARS:
        raise ValueError("Unknown validation year")

    if variant not in FEATURE_VARIANTS:
        raise ValueError("Unknown feature configuration")

    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    if (
        len(features) != len(samples)
        or not features.index.equals(samples.index)
        or features.columns.tolist() != FEATURE_COLUMNS
    ):
        raise ValueError("Feature and sample data do not match")

    folds, holdout_positions = create_chronological_splits(samples)

    matching_folds = [fold for fold in folds if fold["validation_year"] == year]

    if len(matching_folds) != 1:
        raise ValueError("Requested validation fold is missing")

    fold = matching_folds[0]

    training_positions = np.asarray(fold["training_indices"])

    validation_positions = np.asarray(fold["validation_indices"])

    if (
        np.intersect1d(
            training_positions,
            validation_positions,
        ).size
        > 0
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
        raise ValueError("Chronological folds overlap")

    dates = pd.to_datetime(
        samples["target_date_pub"],
        errors="raise",
    )

    training_dates = dates.iloc[training_positions]
    validation_dates = dates.iloc[validation_positions]

    if (
        training_dates.isna().any()
        or validation_dates.isna().any()
        or (training_dates >= HOLDOUT_START).any()
        or (validation_dates >= HOLDOUT_START).any()
        or training_dates.max() >= validation_dates.min()
    ):
        raise ValueError("Invalid chronological validation dates")

    training_features = features.iloc[training_positions]
    validation_features = features.iloc[validation_positions]

    training_targets = samples.iloc[training_positions]["target_beat"].to_numpy()

    validation_targets = samples.iloc[validation_positions]["target_beat"].to_numpy()

    fit_start = perf_counter()

    model = build_model(
        model_name,
        FEATURE_VARIANTS[variant],
    )

    model.fit(training_features, training_targets)

    fit_seconds = perf_counter() - fit_start
    peak_memory_mib = get_peak_memory_mib()

    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Unexpected classifier class ordering")

    prediction_start = perf_counter()

    probabilities = model.predict_proba(validation_features)[:, 1]

    prediction_seconds = perf_counter() - prediction_start

    return {
        "model": model_name,
        "year": year,
        "variant": variant,
        "training_samples": len(training_positions),
        "validation_samples": len(validation_positions),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(prediction_seconds),
        "peak_rss_mib": float(peak_memory_mib),
        "roc_auc": float(roc_auc_score(validation_targets, probabilities)),
        "holdout_used": False,
    }


def summarize_measurements(runs: list[dict]) -> dict:
    """calculates means and standard deviations from repeated runs"""
    if not runs:
        raise ValueError("At least one benchmark run is required")

    summary = {"runs": len(runs)}

    for measurement in BENCHMARK_MEASUREMENTS:
        values = [float(run[measurement]) for run in runs]

        if not all(np.isfinite(value) and value >= 0 for value in values):
            raise ValueError("Invalid benchmark measurements")

        summary[f"mean_{measurement}"] = float(mean(values))

        summary[f"std_{measurement}"] = float(stdev(values)) if len(values) > 1 else 0.0

    return summary


def run_benchmark(
    model_name: str,
    repeats: int = BENCHMARK_REPEATS,
) -> dict:
    """benchmarks both feature configurations on chronological folds"""
    if model_name not in MODEL_NAMES:
        raise ValueError("Unknown classifier")

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("Repeat count must be a positive integer")

    runs = []

    for year in VALIDATION_YEARS:
        for repeat in range(1, repeats + 1):
            variants = list(FEATURE_VARIANTS)

            if repeat % 2 == 0:
                variants.reverse()

            for variant in variants:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "earnings_surprise.evaluation.benchmark_other_classifiers",
                        "--worker",
                        "--model",
                        model_name,
                        "--year",
                        str(year),
                        "--variant",
                        variant,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if completed.returncode != 0:
                    raise RuntimeError(
                        f"Benchmark failed for {model_name}, "
                        f"{year}, {variant}:\n"
                        f"{completed.stderr}"
                    )

                results = [
                    line[len(WORKER_RESULT_PREFIX) :]
                    for line in completed.stdout.splitlines()
                    if line.startswith(WORKER_RESULT_PREFIX)
                ]

                if len(results) != 1:
                    raise RuntimeError("Worker returned an unexpected result")

                result = json.loads(results[0])
                result["repeat"] = repeat

                runs.append(result)

                print(
                    f"{year} | {variant} | repeat {repeat}: "
                    f"fit={result['fit_seconds']:.3f}s, "
                    f"memory={result['peak_rss_mib']:.1f} MiB",
                    flush=True,
                )

    by_year = {}

    for year in VALIDATION_YEARS:
        by_year[str(year)] = {}

        for variant in FEATURE_VARIANTS:
            matching_runs = [
                run for run in runs if run["year"] == year and run["variant"] == variant
            ]

            auc_values = [run["roc_auc"] for run in matching_runs]

            if not np.allclose(
                auc_values,
                auc_values[0],
                rtol=0,
                atol=AUC_TOLERANCE,
            ):
                raise ValueError("Repeated ROC-AUC values do not match")

            by_year[str(year)][variant] = summarize_measurements(matching_runs)

            by_year[str(year)][variant]["roc_auc"] = float(mean(auc_values))

    overall = {}

    for variant in FEATURE_VARIANTS:
        matching_runs = [run for run in runs if run["variant"] == variant]

        overall[variant] = summarize_measurements(matching_runs)

    return {
        "experiment": "repeated_classifier_feature_efficiency",
        "model": model_name,
        "repeats_per_fold": repeats,
        "validation_years": list(VALIDATION_YEARS),
        "holdout_used": False,
        "process_isolation": "fresh worker process per fit",
        "fit_timing": (
            "wall-clock time including model construction, "
            "preprocessing, feature selection, and fitting"
        ),
        "memory_measurement": (
            "peak resident memory at fit completion " "for the entire worker process"
        ),
        "memory_unit": "MiB",
        "runs": runs,
        "by_year": by_year,
        "overall": overall,
    }


def main() -> None:
    """runs a complete benchmark or an independent worker"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_NAMES,
    )

    parser.add_argument(
        "--worker",
        action="store_true",
    )

    parser.add_argument("--year", type=int)

    parser.add_argument(
        "--variant",
        choices=list(FEATURE_VARIANTS),
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=BENCHMARK_REPEATS,
    )

    arguments = parser.parse_args()

    if arguments.worker:
        if arguments.year is None or arguments.variant is None:
            parser.error("Worker requires --year and --variant")

        result = benchmark_one_fit(
            arguments.model,
            arguments.year,
            arguments.variant,
        )

        print(WORKER_RESULT_PREFIX + json.dumps(result))
        return

    report = run_benchmark(
        arguments.model,
        arguments.repeats,
    )

    output_path = RESULTS_DIRECTORY / (f"{arguments.model}_efficiency.json")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nClassifier Efficiency Summary")
    print("-----------------------------")

    for variant, result in report["overall"].items():
        print(
            f"{variant}: "
            f"fit={result['mean_fit_seconds']:.3f}s, "
            f"predict={result['mean_predict_seconds']:.4f}s, "
            f"peak RSS={result['mean_peak_rss_mib']:.1f} MiB"
        )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
