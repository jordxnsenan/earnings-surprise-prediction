import pytest
import matplotlib
matplotlib.use("Agg")

from earnings_surprise.tuning.compare_tuning_results import (
    MODEL_NAMES,
    build_tuning_comparison,
    plot_tuning_comparison,
)

ORIGINAL_METRICS = {
    "macro_f1": 0.60,
    "accuracy": 0.65,
    "roc_auc": 0.70,
    "precision_class_0": 0.55,
    "recall_class_0": 0.40,
    "precision_class_1": 0.70,
    "recall_class_1": 0.80,
}

TUNED_METRICS = {
    "macro_f1": 0.62,
    "accuracy": 0.62,
    "roc_auc": 0.70,
    "precision_class_0": 0.52,
    "recall_class_0": 0.65,
    "precision_class_1": 0.73,
    "recall_class_1": 0.59,
}


def make_test_reports():
    """creates matching synthetic search and assessment reports"""
    searches = {}
    assessments = {}

    parameters = {"classifier__max_depth": 3}

    for name in MODEL_NAMES:
        searches[name] = {
            "model": ("xgboost_all_features" if name == "xgboost" else name),
            "scoring": "f1_macro",
            "iterations": 1,
            "search_samples": 100,
            "search_validation_years": [2020, 2021, 2022],
            "assessment_year": 2023,
            "assessment_samples": 40,
            "holdout_samples": 20,
            "assessment_used": False,
            "holdout_used": False,
            "refit_during_search": False,
            "baseline": {
                "mean_macro_f1": 0.60,
            },
            "best_configuration": {
                "mean_macro_f1": 0.62,
                "improvement_over_baseline": 0.02,
                "pipeline_parameters": parameters,
            },
            "candidates": [{}],
        }

        assessments[name] = {
            "assessment_year": 2023,
            "training_samples": 100,
            "assessment_samples": 40,
            "holdout_samples": 20,
            "holdout_used": False,
            "assessment_used_during_search": False,
            "validation_filtered": False,
            "selected_hyperparameters": parameters,
            "search_mean_macro_f1": 0.62,
            "baseline": ORIGINAL_METRICS.copy(),
            "tuned": TUNED_METRICS.copy(),
            "difference_tuned_minus_baseline": {
                metric: (TUNED_METRICS[metric] - ORIGINAL_METRICS[metric])
                for metric in ORIGINAL_METRICS
            },
        }

        if name != "xgboost":
            assessments[name]["model"] = name

    return searches, assessments


def test_comparison_matches_search_and_assessment_results():
    searches, assessments = make_test_reports()

    summary = build_tuning_comparison(
        searches,
        assessments,
    )

    assert summary["holdout_used"] is False
    assert summary["populations"]["holdout_samples"] == 20
    assert set(summary["models"]) == set(MODEL_NAMES)

    for model in summary["models"].values():
        assert model["search"]["macro_f1_change"] == pytest.approx(0.02)

        assert model["assessment"]["changes"]["macro_f1"] == pytest.approx(0.02)

        assert model["assessment"]["changes"]["accuracy"] == pytest.approx(-0.03)


def test_comparison_creates_figure(tmp_path):
    searches, assessments = make_test_reports()

    summary = build_tuning_comparison(
        searches,
        assessments,
    )

    output_path = tmp_path / "tuning.png"

    plot_tuning_comparison(summary, output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_comparison_rejects_holdout_usage():
    searches, assessments = make_test_reports()

    assessments["decision_tree"]["holdout_used"] = True

    with pytest.raises(ValueError, match="reserved"):
        build_tuning_comparison(
            searches,
            assessments,
        )


def test_comparison_rejects_changed_hyperparameters():
    searches, assessments = make_test_reports()

    assessments["random_forest"]["selected_hyperparameters"] = {
        "classifier__max_depth": 5,
    }

    with pytest.raises(ValueError, match="configuration"):
        build_tuning_comparison(
            searches,
            assessments,
        )


def test_comparison_rejects_mismatched_populations():
    searches, assessments = make_test_reports()

    assessments["decision_tree"]["assessment_samples"] = 41
    searches["decision_tree"]["assessment_samples"] = 41

    with pytest.raises(ValueError, match="populations"):
        build_tuning_comparison(
            searches,
            assessments,
        )


def test_comparison_rejects_inconsistent_metrics():
    searches, assessments = make_test_reports()

    assessments["logistic_regression"]["tuned"]["macro_f1"] = 0.99

    with pytest.raises(ValueError, match="metrics"):
        build_tuning_comparison(
            searches,
            assessments,
        )


def test_comparison_rejects_mismatched_search_years():
    searches, assessments = make_test_reports()

    searches["xgboost"]["search_validation_years"] = [
        2020,
        2021,
        2023,
    ]

    with pytest.raises(ValueError, match="validation periods"):
        build_tuning_comparison(
            searches,
            assessments,
        )


def test_comparison_rejects_missing_model():
    searches, assessments = make_test_reports()

    del searches["random_forest"]

    with pytest.raises(ValueError, match="Missing classifier"):
        build_tuning_comparison(
            searches,
            assessments,
        )
