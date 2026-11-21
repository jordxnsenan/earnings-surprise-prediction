import pandas as pd
import pytest

from earnings_surprise.data.audit_snapshots import (
    summarize_date_relationships,
    summarize_snapshot_repetition,
    find_date_examples
)


def test_summarize_snapshot_repetition():
    data = pd.DataFrame({
        "ticker": ["A", "A", "A", "B", "B", "C", "C", "D"],
        "date": [
            "2024-01-01", "2024-01-01", "2024-01-08",
            "2024-02-01", "2024-02-08",
            "2024-03-01", "2024-03-08", "2024-04-01",
        ],
        "date_pub": [
            "2024-02-01", "2024-02-01", "2024-02-01",
            "2024-03-01", "2024-03-01",
            "2024-04-01", "2024-04-01", None,
        ],
        "actual_earning_result": [
            1.0, 1.0, 1.0, 0.5, 0.6, None, 0.5, 0.5
        ],
        "estimated_earning": [
            0.8, 0.8, 0.9, 0.3, 0.3, 0.4, 0.4, 0.4
        ],
        "eps_surprise": [
            0.2, 0.2, 0.1, 0.2, 0.3, None, 0.1, 0.1
        ],
    })

    original_data = data.copy(deep=True)

    summary = summarize_snapshot_repetition(data)

    assert summary["total_rows"] == 8
    assert summary["rows_with_candidate_key"] == 7
    assert summary["rows_without_candidate_key"] == 1

    assert summary["candidate_key_count"] == 3
    assert summary["keys_with_multiple_rows"] == 3
    assert summary["rows_in_repeated_keys"] == 7
    assert summary["extra_rows_from_repetition"] == 4

    assert summary["exact_duplicate_rows"] == 1
    assert summary["maximum_rows_per_key"] == 3

    conflicts = summary["keys_with_conflicting_financial_values"]

    assert conflicts["actual_earning_result"] == 1
    assert conflicts["estimated_earning"] == 1
    assert conflicts["eps_surprise"] == 2

    # Auditing must not modify the original data.
    pd.testing.assert_frame_equal(data, original_data)


def test_missing_required_columns():
    data = pd.DataFrame({
        "ticker": ["A"],
        "date_pub": ["2024-02-01"],
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        summarize_snapshot_repetition(data)


def test_no_valid_candidate_keys():
    data = pd.DataFrame({
        "ticker": ["A"],
        "date_pub": [None],
        "actual_earning_result": [0.5],
        "estimated_earning": [0.4],
        "eps_surprise": [0.1],
    })

    summary = summarize_snapshot_repetition(data)

    assert summary["candidate_key_count"] == 0
    assert summary["keys_with_multiple_rows"] == 0
    assert summary["maximum_rows_per_key"] == 0


def test_summarize_date_relationships():
    data = pd.DataFrame({
        "ticker": ["A", "A", "B", "C", "D", "E"],
        "date": [
            "2024-02-01",
            "2024-02-10",
            "2024-03-03",
            "2024-04-01",
            None,
            "invalid",
        ],
        "date_pub": [
            "2024-02-05",
            "2024-02-10",
            "2024-03-01",
            None,
            "2024-04-10",
            "2024-05-01",
        ],
        "actual_earning_result": [
            0.5, 0.6, None, 0.8, 0.2, 0.3
        ],
    })

    original_data = data.copy(deep=True)

    summary = summarize_date_relationships(data)

    assert summary["rows_with_both_dates"] == 3
    assert summary["rows_missing_or_invalid_either_date"] == 3

    assert summary["rows_before_publication"] == 1
    assert summary["rows_on_publication_date"] == 1
    assert summary["rows_after_publication"] == 1

    assert summary["prepublication_rows_with_recorded_actual_eps"] == 1
    assert summary["candidate_events_with_prepublication_actual_eps"] == 1

    day_gaps = summary["publication_minus_snapshot_days"]

    assert day_gaps["minimum"] == -2
    assert day_gaps["median"] == 0
    assert day_gaps["maximum"] == 4

    pd.testing.assert_frame_equal(data, original_data)


def test_date_relationships_requires_columns():
    data = pd.DataFrame({
        "ticker": ["A"],
        "date": ["2024-01-01"],
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        summarize_date_relationships(data)
        

def test_find_date_examples():
    data = pd.DataFrame({
        "ticker": ["A", "A", "B", "C", "D", "E"],
        "date": [
            "2024-01-01",
            "2024-01-04",
            "2024-02-01",
            "2024-03-05",
            "2024-04-01",
            "2024-05-01",
        ],
        "date_pub": [
            "2024-01-10",
            "2024-01-10",
            "2024-02-05",
            "2024-03-02",
            "2024-04-01",
            "2024-05-20",
        ],
        "actual_earning_result": [
            0.5, 0.5, 0.1, 0.2, 0.3, None
        ],
        "estimated_earning": [
            0.4, 0.4, 0.1, 0.2, 0.3, 0.3
        ],
        "eps_surprise": [
            0.1, 0.1, 0.0, 0.0, 0.0, None
        ],
    })

    original_data = data.copy(deep=True)

    examples = find_date_examples(data, example_count=2)

    longest_gaps = examples["largest_prepublication_gaps"]
    postpublication = examples["postpublication_examples"]

    assert longest_gaps["ticker"].tolist() == ["A", "B"]

    assert longest_gaps["days_before_publication"].tolist() == [
        9, 4
    ]

    assert postpublication["ticker"].tolist() == ["C"]
    assert postpublication["days_before_publication"].iloc[0] == -3

    pd.testing.assert_frame_equal(data, original_data)