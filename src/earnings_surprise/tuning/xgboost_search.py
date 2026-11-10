import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.features import (
    FEATURE_COLUMNS,
    FEATURE_MATRIX_PATH,
)
from earnings_surprise.models.xgboost_model import (
    build_xgboost_pipeline,
)
from earnings_surprise.tuning.splits import (
    ASSESSMENT_YEAR,
    SEARCH_VALIDATION_YEARS,
    build_tuning_partitions,
)

SEARCH_ITERATIONS = 16
SEARCH_RANDOM_STATE = 42
SEARCH_SCORING = "f1_macro"

XGBOOST_PARAMETERS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.03, 0.05, 0.10, 0.15],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_lambda": [1.0, 5.0],
}

SEARCH_RESULTS_PATH = Path("results/tuning/xgboost_random_search.json")


def build_search_space(
    pipeline: Pipeline,
) -> tuple[dict, str]:
    """builds parameter names for the existing XGBoost pipeline"""
    if not isinstance(pipeline, Pipeline):
        raise ValueError("Expected an sklearn Pipeline")

    estimator_name, estimator = pipeline.steps[-1]

    available_parameters = estimator.get_params()

    missing_parameters = set(XGBOOST_PARAMETERS) - set(available_parameters)

    if missing_parameters:
        raise ValueError(f"Missing XGBoost parameters: {sorted(missing_parameters)}")

    parameter_space = {
        f"{estimator_name}__{name}": values
        for name, values in XGBOOST_PARAMETERS.items()
    }

    return parameter_space, estimator_name


def summarize_search_candidates(
    search: RandomizedSearchCV,
    validation_years: tuple[int, ...],
    estimator_name: str,
) -> list[dict]:
    """preserves candidate parameters and chronological fold scores"""
    results = search.cv_results_
    prefix = f"{estimator_name}__"

    candidates = []

    for index, parameters in enumerate(results["params"]):
        yearly_scores = {
            str(year): float(results[f"split{fold_index}_test_score"][index])
            for fold_index, year in enumerate(validation_years)
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
                "std_macro_f1": float(results["std_test_score"][index]),
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


def run_xgboost_search(
    features: pd.DataFrame,
    samples: pd.DataFrame,
    iterations: int = SEARCH_ITERATIONS,
) -> dict:
    """searches XGBoost hyperparameters using chronological training folds"""
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

    if "target_beat" not in samples.columns:
        raise ValueError("Missing classification target")

    partitions = build_tuning_partitions(samples)

    search_positions = partitions["search_positions"]

    search_features = features.iloc[search_positions]
    search_targets = samples.iloc[search_positions]["target_beat"].to_numpy()

    search_dates = pd.to_datetime(samples.iloc[search_positions]["target_date_pub"])

    if search_dates.isna().any() or (search_dates.dt.year >= ASSESSMENT_YEAR).any():
        raise ValueError("Assessment data entered hyperparameter search")

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

    pipeline = build_xgboost_pipeline(selected_feature_count=None)

    parameter_space, estimator_name = build_search_space(pipeline)

    # establishes the original model's score on the same search folds
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
        return_train_score=False,
        error_score="raise",
    )

    search.fit(
        search_features,
        search_targets,
    )

    candidates = summarize_search_candidates(
        search,
        validation_years,
        estimator_name,
    )

    best_parameters = {name: value for name, value in search.best_params_.items()}

    best_model_parameters = {
        name.removeprefix(f"{estimator_name}__"): value
        for name, value in best_parameters.items()
    }

    baseline_mean = float(np.mean(baseline_scores))
    best_mean = float(search.best_score_)

    return {
        "experiment": "xgboost_chronological_random_search",
        "model": "xgboost_all_features",
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
        "search_space": XGBOOST_PARAMETERS,
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
            "mean_macro_f1": best_mean,
            "improvement_over_baseline": (best_mean - baseline_mean),
            "pipeline_parameters": best_parameters,
            "model_parameters": best_model_parameters,
        },
        "candidates": candidates,
    }


def main() -> None:
    """runs the XGBoost search and save its results"""
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    report = run_xgboost_search(
        features,
        samples,
    )

    SEARCH_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SEARCH_RESULTS_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    baseline = report["baseline"]
    best = report["best_configuration"]

    print("XGBoost Chronological Hyperparameter Search")
    print("-------------------------------------------")

    print(f"Search observations: {report['search_samples']:,}")
    print(f"Candidate configurations: {report['iterations']}")
    print(f"Validation years: {report['search_validation_years']}")

    print(f"\nOriginal mean macro F1: " f"{baseline['mean_macro_f1']:.5f}")

    print(f"Best mean macro F1: " f"{best['mean_macro_f1']:.5f}")

    print("Improvement: " f"{best['improvement_over_baseline']:+.5f}")

    print("\nBest XGBoost parameters:")

    for name, value in best["model_parameters"].items():
        print(f"  {name}: {value}")

    print("\nTop five candidates:")

    for candidate in report["candidates"][:5]:
        print(
            f"  Rank {candidate['rank']}: " f"macro F1={candidate['mean_macro_f1']:.5f}"
        )

    print(f"\nResults saved to: {SEARCH_RESULTS_PATH}")


if __name__ == "__main__":
    main()
