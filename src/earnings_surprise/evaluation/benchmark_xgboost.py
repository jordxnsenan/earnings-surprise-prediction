import argparse
import json
import numpy as np
import pandas as pd
import platform
import resource
import subprocess
import sys

from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
from sklearn.metrics import roc_auc_score

from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.roc_comparison import (
    MODEL_VARIANTS,
    ROC_RESULTS_PATH,
)
from earnings_surprise.evaluation.splits import (
    HOLDOUT_START,
    VALIDATION_YEARS,
    create_chronological_splits,
)
from earnings_surprise.features import FEATURE_MATRIX_PATH
from earnings_surprise.models.xgboost_model import (
    build_xgboost_pipeline,
)

BENCHMARK_REPEATS = 3
RESULT_PREFIX = "BENCHMARK_RESULT:"
MEASUREMENTS = (
    "fit_seconds",
    "predict_seconds",
    "peak_rss_mib",
)

BENCHMARK_PATH = Path("results/classification/xgboost_efficiency.json")


def peak_process_rss_mib() -> float:
    """returns the worker's peak resident memory in MiB"""
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # reports bytes for macOS wand KiB for Linux
    if platform.system() == "Darwin":
        return float(peak_rss / (1024 * 1024))

    return float(peak_rss / 1024)


def benchmark_one_fit(year: int, variant: str) -> dict:
    """benchmarks one XGBoost variant in an independent process"""
    if year not in VALIDATION_YEARS or variant not in MODEL_VARIANTS:
        raise ValueError("Unknown benchmark year or model variant")

    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    folds, holdout_positions = create_chronological_splits(samples)

    matching_folds = [fold for fold in folds if fold["validation_year"] == year]

    if len(matching_folds) != 1:
        raise ValueError("Requested chronological fold is missing")

    fold = matching_folds[0]

    training_positions = np.asarray(fold["training_indices"])
    validation_positions = np.asarray(fold["validation_indices"])

    evaluated_positions = np.concatenate(
        [
            training_positions,
            validation_positions,
        ]
    )

    dates = pd.to_datetime(samples["target_date_pub"])

    if (
        dates.iloc[evaluated_positions].isna().any()
        or (dates.iloc[evaluated_positions] >= HOLDOUT_START).any()
        or dates.iloc[training_positions].max()
        >= dates.iloc[validation_positions].min()
        or np.intersect1d(
            evaluated_positions,
            holdout_positions,
        ).size
        > 0
    ):
        raise ValueError("Invalid chronological validation split")

    training_features = features.iloc[training_positions]
    validation_features = features.iloc[validation_positions]

    training_targets = samples.iloc[training_positions]["target_beat"].to_numpy()

    validation_targets = samples.iloc[validation_positions]["target_beat"].to_numpy()

    fit_start = perf_counter()

    model = build_xgboost_pipeline(
        selected_feature_count=MODEL_VARIANTS[variant],
    )

    model.fit(training_features, training_targets)

    fit_seconds = perf_counter() - fit_start
    peak_rss_mib = peak_process_rss_mib()

    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Unexpected classifier class ordering")

    prediction_start = perf_counter()

    probabilities = model.predict_proba(validation_features)[:, 1]

    prediction_seconds = perf_counter() - prediction_start

    return {
        "year": int(year),
        "variant": variant,
        "training_samples": int(len(training_features)),
        "validation_samples": int(len(validation_features)),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(prediction_seconds),
        "peak_rss_mib": peak_rss_mib,
        "roc_auc": float(roc_auc_score(validation_targets, probabilities)),
        "holdout_used": False,
    }


def summarize_measurements(runs: list[dict]) -> dict:
    """summarizes repeated runtime and memory measurements"""
    if not runs:
        raise ValueError("At least one benchmark run is required")

    summary = {"runs": len(runs)}

    for measurement in MEASUREMENTS:
        values = [float(run[measurement]) for run in runs]

        if not all(np.isfinite(value) and value >= 0 for value in values):
            raise ValueError("Benchmark measurements must be nonnegative and finite")

        summary[f"mean_{measurement}"] = float(mean(values))

        summary[f"std_{measurement}"] = float(stdev(values)) if len(values) > 1 else 0.0

    return summary


def verify_roc_results(
    runs: list[dict],
    reference: dict,
) -> None:
    """checks that benchmark models reproduce saved ROC-AUC values"""
    expected = {
        (fold["validation_year"], variant): result["roc_auc"]
        for fold in reference["folds"]
        for variant, result in fold["variants"].items()
    }

    for run in runs:
        key = (run["year"], run["variant"])

        if key not in expected or not np.isclose(
            run["roc_auc"],
            expected[key],
            rtol=0,
            atol=1e-6,
        ):
            raise ValueError("Benchmark ROC-AUC differs from saved comparison")


def run_benchmark(
    repeats: int = BENCHMARK_REPEATS,
) -> dict:
    """collects independent measurements across validation folds"""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("Repeat count must be a positive integer")

    runs = []

    for year in VALIDATION_YEARS:
        for repeat in range(1, repeats + 1):
            variants = list(MODEL_VARIANTS)

            # alternates the order to reduce systematic run-order effects
            if repeat % 2 == 0:
                variants.reverse()

            for variant in variants:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "earnings_surprise.evaluation.benchmark_xgboost",
                        "--worker",
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
                        f"Benchmark failed for {year}, {variant}:\n"
                        f"{completed.stderr}"
                    )

                messages = [
                    line[len(RESULT_PREFIX) :]
                    for line in completed.stdout.splitlines()
                    if line.startswith(RESULT_PREFIX)
                ]

                if len(messages) != 1:
                    raise RuntimeError("Worker did not return one benchmark result")

                result = json.loads(messages[0])
                result["repeat"] = repeat

                runs.append(result)

                print(
                    f"{year} | {variant} | repeat {repeat}: "
                    f"fit={result['fit_seconds']:.3f}s, "
                    f"peak RSS={result['peak_rss_mib']:.1f} MiB"
                )

    with ROC_RESULTS_PATH.open(encoding="utf-8") as file:
        reference = json.load(file)

    verify_roc_results(runs, reference)

    by_year = {
        str(year): {
            variant: summarize_measurements(
                [
                    run
                    for run in runs
                    if run["year"] == year and run["variant"] == variant
                ]
            )
            for variant in MODEL_VARIANTS
        }
        for year in VALIDATION_YEARS
    }

    overall = {
        variant: summarize_measurements(
            [run for run in runs if run["variant"] == variant]
        )
        for variant in MODEL_VARIANTS
    }

    return {
        "experiment": "xgboost_repeated_feature_efficiency",
        "repeats_per_fold": repeats,
        "holdout_used": False,
        "process_isolation": "fresh worker process for each fit",
        "fit_timing": (
            "wall-clock time including preprocessing, "
            "feature selection, and model fitting"
        ),
        "memory_measurement": (
            "peak resident memory of the entire worker process "
            "at fit completion; includes dataset and libraries"
        ),
        "memory_unit": "MiB",
        "runs": runs,
        "by_year": by_year,
        "overall": overall,
    }


def main() -> None:
    """runs the benchmark or one independent worker"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--year", type=int)
    parser.add_argument(
        "--variant",
        choices=list(MODEL_VARIANTS),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=BENCHMARK_REPEATS,
    )

    args = parser.parse_args()

    if args.worker:
        if args.year is None or args.variant is None:
            parser.error("Workers require --year and --variant")

        result = benchmark_one_fit(
            args.year,
            args.variant,
        )

        print(RESULT_PREFIX + json.dumps(result))
        return

    report = run_benchmark(repeats=args.repeats)

    BENCHMARK_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    BENCHMARK_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nRepeated XGBoost Efficiency Comparison")
    print("--------------------------------------")

    for variant, result in report["overall"].items():
        print(
            f"{variant}: "
            f"fit={result['mean_fit_seconds']:.3f}s, "
            f"predict={result['mean_predict_seconds']:.4f}s, "
            f"peak RSS={result['mean_peak_rss_mib']:.1f} MiB"
        )

    print(f"\nResults saved to: {BENCHMARK_PATH}")


if __name__ == "__main__":
    main()
