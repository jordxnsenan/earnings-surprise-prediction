import pytest

from earnings_surprise.evaluation.metrics import (
    evaluate_predictions,
)


def test_evaluate_predictions():
    actual = [1, 1, 0, 0, 1, 0]
    predicted = [1, 0, 1, 0, 1, 0]

    probabilities = [0.9, 0.4, 0.7, 0.2, 0.8, 0.1]

    results = evaluate_predictions(
        actual,
        predicted,
        probabilities,
    )

    assert results["samples"] == 6

    assert results["accuracy"] == pytest.approx(2 / 3)
    assert results["macro_f1"] == pytest.approx(2 / 3)
    assert results["roc_auc"] == pytest.approx(8 / 9)

    assert results["confusion_matrix"] == [
        [2, 1],
        [1, 2],
    ]

    beat_metrics = results["per_class"]["beat"]

    assert beat_metrics["precision"] == pytest.approx(2 / 3)
    assert beat_metrics["recall"] == pytest.approx(2 / 3)
    assert beat_metrics["f1"] == pytest.approx(2 / 3)
    assert beat_metrics["support"] == 3

    miss_metrics = results["per_class"]["meet_or_miss"]

    assert miss_metrics["support"] == 3


def test_evaluate_predictions_without_probabilities():
    actual = [1, 0, 1, 0]
    predicted = [1, 0, 0, 0]

    results = evaluate_predictions(actual, predicted)

    assert results["accuracy"] == pytest.approx(0.75)
    assert results["roc_auc"] is None


def test_evaluate_predictions_with_single_class():
    actual = [1, 1, 1]
    predicted = [1, 1, 1]
    probabilities = [0.8, 0.7, 0.9]

    results = evaluate_predictions(
        actual,
        predicted,
        probabilities,
    )

    assert results["accuracy"] == 1.0
    assert results["macro_f1"] == pytest.approx(0.5)
    assert results["roc_auc"] is None

    assert results["confusion_matrix"] == [
        [0, 0],
        [0, 3],
    ]


def test_evaluate_predictions_rejects_invalid_labels():
    with pytest.raises(ValueError, match="Actual labels"):
        evaluate_predictions([0, 2], [0, 1])

    with pytest.raises(ValueError, match="Predictions"):
        evaluate_predictions([0, 1], [0, -1])


def test_evaluate_predictions_requires_matching_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_predictions([0, 1, 1], [0, 1])


@pytest.mark.parametrize(
    "invalid_probabilities",
    [
        [0.5, float("nan")],
        [0.5, float("inf")],
        [-0.1, 0.5],
        [0.5, 1.2],
    ],
)
def test_evaluate_predictions_rejects_invalid_probabilities(
    invalid_probabilities,
):
    with pytest.raises(ValueError, match="Probabilities"):
        evaluate_predictions(
            [0, 1],
            [0, 1],
            invalid_probabilities,
        )