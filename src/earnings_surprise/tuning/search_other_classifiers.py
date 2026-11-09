import argparse
import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
)

from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.evaluation.benchmark_other_classifiers import (
    MODEL_NAMES,
    build_model,
)
from earnings_surprise.features import (
    FEATURE_COLUMNS,
    FEATURE_MATRIX_PATH,
)
from earnings_surprise.tuning.splits import (
    ASSESSMENT_YEAR,
    SEARCH_VALIDATION_YEARS,
    build_tuning_partitions,
)

SEARCH_SCORING = "f1_macro"
SEARCH_RANDOM_STATE = 42

SEARCH_ITERATIONS = {
    "random_forest": 16,
    "logistic_regression": 10,
    "decision_tree": 16,
}

SEARCH_PARAMETERS = {
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 8, 12, None],
        "min_samples_leaf": [1, 5, 20],
        "max_features": ["sqrt", 0.75],
        "class_weight": [None, "balanced"],
    },
    "logistic_regression": {
        "C": [0.1, 0.3, 1.0, 3.0, 10.0],
        "class_weight": [None, "balanced"],
    },
    "decision_tree": {
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_leaf": [5, 20, 50],
        "class_weight": [None, "balanced"],
    },
}

RESULTS_DIRECTORY = Path("results/tuning")


def build_search_space(
    pipeline: Pipeline,
    model_name: str,
) -> tuple[dict, str]:
    """creates parameter names for an existing classifier pipeline"""
    if model_name not in SEARCH_PARAMETERS:
        raise ValueError("Unknown classifier")

    if not isinstance(pipeline, Pipeline):
        raise ValueError("Expected an sklearn Pipeline")

    estimator_name, estimator = pipeline.steps[-1]
    available_parameters = estimator.get_params()

    parameters = SEARCH_PARAMETERS[model_name]

    if not set(parameters).issubset(available_parameters):
        raise ValueError("Search contains unsupported model parameters")

    search_space = {
        f"{estimator_name}__{name}": values for name, values in parameters.items()
    }

    return search_space, estimator_name


def summarize_candidates(
    search: RandomizedSearchCV,
    estimator_name: str,
) -> list[dict]:
    """records every candidate's parameters and yearly validation scores"""
    results = search.cv_results_
    prefix = f"{estimator_name}__"

    candidates = []

    for index, parameters in enumerate(results["params"]):
        yearly_scores = {
            str(year): float(results[f"split{fold_index}_test_score"][index])
            for fold_index, year in enumerate(SEARCH_VALIDATION_YEARS)
        }

        candidates.append(
            {
                "candidate_number": index + 1,
                "rank": int(results["rank_test_score"][index]),
                "parameters": {
                    name.removeprefix(prefix): value
                    for name, value in parameters.items()
                },
                "mean_macro_f1": float(results["mean_test_score"][index]),
                "yearly_macro_f1": yearly_scores,
                "mean_fit_seconds": float(results["mean_fit_time"][index]),
            }
        )

    candidates.sort(
        key=lambda candidate: (
            candidate["rank"],
            candidate["candidate_number"],
        )
    )

    return candidates


def run_classifier_search(
    features: pd.DataFrame,
    samples: pd.DataFrame,
    model_name: str,
    iterations: int | None = None,
) -> dict:
    """tunes a classifier using expanding chronological validation"""
    if model_name not in MODEL_NAMES:
        raise ValueError("Unknown classifier")

    if iterations is None:
        iterations = SEARCH_ITERATIONS[model_name]

    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations < 1
    ):
        raise ValueError("Search iterations must be a positive integer")

    if (
        len(features) != len(samples)
        or not features.index.equals(samples.index)
        or features.columns.tolist() != FEATURE_COLUMNS
    ):
        raise ValueError("Feature and sample data do not match")

    partitions = build_tuning_partitions(samples)

    search_positions = partitions["search_positions"]

    search_features = features.iloc[search_positions]
    search_targets = samples.iloc[search_positions]["target_beat"].to_numpy()

    search_dates = pd.to_datetime(samples.iloc[search_positions]["target_date_pub"])

    if search_dates.isna().any() or (search_dates.dt.year >= ASSESSMENT_YEAR).any():
        raise ValueError("Assessment data entered search")

    if np.intersect1d(
        search_positions,
        partitions["assessment_validation_positions"],
    ).size:
        raise ValueError("Assessment observations entered search")

    if np.intersect1d(
        search_positions,
        partitions["holdout_positions"],
    ).size:
        raise ValueError("Holdout observations entered search")

    validation_years = tuple(
        fold["validation_year"] for fold in partitions["search_folds"]
    )

    if validation_years != SEARCH_VALIDATION_YEARS:
        raise ValueError("Unexpected search validation years")

    cross_validation = [
        (
            fold["training_indices"],
            fold["validation_indices"],
        )
        for fold in partitions["search_folds"]
    ]

    pipeline = build_model(
        model_name,
        selected_feature_count=None,
    )

    parameter_space, estimator_name = build_search_space(
        pipeline,
        model_name,
    )

    baseline_scores = cross_val_score(
        pipeline,
        search_features,
        search_targets,
        cv=cross_validation,
        scoring=SEARCH_SCORING,
        n_jobs=1,
        error_score="raise",
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_space,
        n_iter=iterations,
        scoring=SEARCH_SCORING,
        cv=cross_validation,
        random_state=SEARCH_RANDOM_STATE,
        n_jobs=1,
        refit=False,
        error_score="raise",
    )

    search.fit(search_features, search_targets)

    candidates = summarize_candidates(
        search,
        estimator_name,
    )

    best_parameters = search.best_params_

    model_parameters = {
        name.removeprefix(f"{estimator_name}__"): value
        for name, value in best_parameters.items()
    }

    baseline_mean = float(np.mean(baseline_scores))
    tuned_mean = float(search.best_score_)

    return {
        "experiment": "chronological_classifier_random_search",
        "model": model_name,
        "scoring": SEARCH_SCORING,
        "random_state": SEARCH_RANDOM_STATE,
        "iterations": iterations,
        "search_samples": len(search_positions),
        "search_validation_years": list(validation_years),
        "assessment_year": ASSESSMENT_YEAR,
        "assessment_samples": len(partitions["assessment_validation_positions"]),
        "holdout_samples": len(partitions["holdout_positions"]),
        "assessment_used": False,
        "holdout_used": False,
        "refit_during_search": False,
        "search_space": SEARCH_PARAMETERS[model_name],
        "baseline": {
            "mean_macro_f1": baseline_mean,
            "yearly_macro_f1": {
                str(year): float(score)
                for year, score in zip(
                    validation_years,
                    baseline_scores,
                    strict=True,
                )
            },
        },
        "best_configuration": {
            "mean_macro_f1": tuned_mean,
            "improvement_over_baseline": (tuned_mean - baseline_mean),
            "pipeline_parameters": best_parameters,
            "model_parameters": model_parameters,
        },
        "candidates": candidates,
    }


def main() -> None:
    """runs a chronological hyperparameter search"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_NAMES,
    )

    args = parser.parse_args()

    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    report = run_classifier_search(
        features,
        samples,
        model_name=args.model,
    )

    output_path = RESULTS_DIRECTORY / f"{args.model}_random_search.json"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"{args.model} — Chronological Parameter Search")
    print("---------------------------------------------")

    print(f"Search samples: {report['search_samples']:,}")
    print(f"Candidates: {len(report['candidates'])}")

    print("Baseline mean macro F1: " f"{report['baseline']['mean_macro_f1']:.5f}")

    best = report["best_configuration"]

    print(f"Best mean macro F1: {best['mean_macro_f1']:.5f}")
    print("Improvement: " f"{best['improvement_over_baseline']:+.5f}")

    print("\nBest parameters:")

    for name, value in best["model_parameters"].items():
        print(f"  {name}: {value}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
