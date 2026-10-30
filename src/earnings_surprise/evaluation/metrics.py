
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


CLASS_LABELS = [0, 1]

CLASS_NAMES = {
    0: "meet_or_miss",
    1: "beat",
}


def _validate_binary_labels(values, description: str) -> np.ndarray:
    """validate binary classification labels or predictions"""
    array = np.asarray(values)

    if array.ndim != 1:
        raise ValueError(f"{description} must be one-dimensional")

    labels = pd.Series(array)

    if labels.empty:
        raise ValueError(f"{description} cannot be empty")

    if labels.isna().any() or not labels.isin(CLASS_LABELS).all():
        raise ValueError(f"{description} must contain only 0 or 1")

    return labels.to_numpy(dtype=int)


def _validate_beat_probabilities(
    values,
    expected_count: int,
) -> np.ndarray:
    """validates predicted earnings-beat probabilities"""
    array = np.asarray(values)

    if array.ndim != 1 or len(array) != expected_count:
        raise ValueError("Probability count does not match predictions")

    probabilities = pd.to_numeric(
        pd.Series(array),
        errors="coerce",
    ).to_numpy(dtype=float, na_value=np.nan)

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must contain finite values")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1")

    return probabilities


def evaluate_predictions(
    y_true,
    y_pred,
    beat_probabilities=None,
) -> dict:
    """calculates binary classification evaluation metrics"""
    actual = _validate_binary_labels(y_true, "Actual labels")
    predicted = _validate_binary_labels(y_pred, "Predictions")

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual labels and predictions must have equal lengths"
        )

    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=CLASS_LABELS,
        zero_division=0,
    )

    class_metrics = {}

    for index, class_label in enumerate(CLASS_LABELS):
        class_name = CLASS_NAMES[class_label]

        class_metrics[class_name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }

    roc_auc = None

    if beat_probabilities is not None:
        probabilities = _validate_beat_probabilities(
            beat_probabilities,
            len(actual),
        )

        # ROC-AUC requires the presence of both target classes
        if len(np.unique(actual)) == len(CLASS_LABELS):
            roc_auc = float(
                roc_auc_score(actual, probabilities)
            )

    return {
        "samples": len(actual),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1.mean()),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(
            actual,
            predicted,
            labels=CLASS_LABELS,
        ).tolist(),
        "per_class": class_metrics,
    }