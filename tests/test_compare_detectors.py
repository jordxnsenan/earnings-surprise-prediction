import numpy as np
import pandas as pd
import pytest

from earnings_surprise.outliers.compare_detectors import (
    summarize_detector_agreement,
    verify_saved_lof_assignments,
)


def test_detector_agreement_counts():
    isolation_labels = np.array(
        [
            -1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            1,
        ]
    )

    lof_labels = np.array(
        [
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            1,
        ]
    )

    result = summarize_detector_agreement(
        isolation_labels,
        lof_labels,
    )

    assert result["samples"] == 8
    assert result["isolation_outliers"] == 3
    assert result["lof_outliers"] == 2

    assert result["both_outliers"] == 1
    assert result["isolation_only"] == 2
    assert result["lof_only"] == 1
    assert result["neither"] == 4

    assert result["union_outliers"] == 4
    assert result["jaccard_similarity"] == pytest.approx(0.25)

    assert result["expected_overlap_under_random_selection"] == pytest.approx(0.75)


def test_detector_agreement_handles_no_outliers():
    labels = np.ones(10, dtype=int)

    result = summarize_detector_agreement(labels, labels)

    assert result["both_outliers"] == 0
    assert result["neither"] == 10
    assert result["jaccard_similarity"] is None


def test_detector_agreement_rejects_different_lengths():
    with pytest.raises(ValueError, match="matching lengths"):
        summarize_detector_agreement(
            np.array([-1, 1, 1]),
            np.array([-1, 1]),
        )


def test_detector_agreement_rejects_invalid_labels():
    with pytest.raises(ValueError, match="Invalid outlier labels"):
        summarize_detector_agreement(
            np.array([-1, 0, 1]),
            np.array([-1, 1, 1]),
        )


def make_saved_assignments():
    positions = np.array([2, 5, 8])
    labels = np.array([1, -1, 1])
    scores = np.array([-1.1, -2.2, -1.0])

    saved = pd.DataFrame(
        {
            "source_row_position": positions,
            "anomaly_label": labels,
            "anomaly_score": scores,
        }
    )

    return saved, positions, labels, scores


def test_saved_lof_assignments_match():
    saved, positions, labels, scores = make_saved_assignments()

    verify_saved_lof_assignments(saved, positions, labels, scores)


def test_saved_lof_assignments_reject_wrong_positions():
    saved, positions, labels, scores = make_saved_assignments()

    saved.loc[0, "source_row_position"] = 99

    with pytest.raises(ValueError, match="positions do not match"):
        verify_saved_lof_assignments(saved, positions, labels, scores)


def test_saved_lof_assignments_reject_changed_scores():
    saved, positions, labels, scores = make_saved_assignments()

    saved.loc[0, "anomaly_score"] = -5.0

    with pytest.raises(ValueError, match="scores changed"):
        verify_saved_lof_assignments(saved, positions, labels, scores)
