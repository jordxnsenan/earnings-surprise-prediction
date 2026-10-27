import pandas as pd
import pytest

from earnings_surprise.data.events import (
    build_candidate_event_ledger,
    summarize_event_quality,
)


def make_sample_data() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["A", "A", "A", "B", "B", "C"],
        "date": pd.to_datetime([
            "2024-01-01",
            "2024-01-08",
            "2024-04-01",
            "2024-01-04",
            "2024-01-11",
            "2024-06-01",
        ]),
        "date_pub": pd.to_datetime([
            "2024-02-01",
            "2024-02-01",
            "2024-05-01",
            "2024-03-01",
            "2024-03-01",
            None,
        ]),
        "actual_earning_result": [
            None, 0.5, None, -0.3, -0.3, 0.1
        ],
        "estimated_earning": [
            0.4, 0.4, 0.6, -0.2, -0.2, 0.05
        ],
        "eps_surprise": [
            None, 0.1, None, -0.1, -0.1, 0.05
        ],
    })


def test_build_candidate_event_ledger():
    data = make_sample_data()
    original_data = data.copy(deep=True)

    events, summary = build_candidate_event_ledger(data)

    assert summary["raw_rows"] == 6
    assert summary["rows_with_candidate_key"] == 5
    assert summary["rows_without_candidate_key"] == 1
    assert summary["candidate_event_count"] == 3
    assert summary["events_with_multiple_source_rows"] == 2
    assert summary["events_missing_actual_eps"] == 1

    first_event = events.loc[
        (events["ticker"] == "A")
        & (events["date_pub"] == pd.Timestamp("2024-02-01"))
    ].iloc[0]

    assert first_event["source_row_count"] == 2
    assert first_event["actual_earning_result"] == 0.5
    assert first_event["actual_earning_result_observed_rows"] == 1

    # the later A event must remain even without actual EPS
    missing_event = events.loc[
        (events["ticker"] == "A")
        & (events["date_pub"] == pd.Timestamp("2024-05-01"))
    ].iloc[0]

    assert pd.isna(missing_event["actual_earning_result"])

    pd.testing.assert_frame_equal(data, original_data)


def test_conflicting_values_are_rejected():
    data = make_sample_data()

    data.loc[4, "actual_earning_result"] = -0.25

    with pytest.raises(
        ValueError,
        match="Conflicting non-missing financial values",
    ):
        build_candidate_event_ledger(data)


def test_missing_required_columns():
    data = make_sample_data().drop(columns=["eps_surprise"])

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        build_candidate_event_ledger(data)
        


def test_summarize_event_quality():
    events = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "date_pub": pd.to_datetime([
            "2024-01-08",  # mon.
            "2024-01-13",  # sat.
            "2024-01-14",  # sun.
            "2024-01-15",  # mon.
        ]),
        "actual_earning_result": [
            0.6, 0.6, None, 0.7
        ],
        "estimated_earning": [
            0.4, 0.4, 0.2, 0.6
        ],
        "eps_surprise": [
            0.2, 0.1, None, 0.1
        ],
    })

    original_events = events.copy(deep=True)

    summary = summarize_event_quality(events)

    assert summary["total_events"] == 4
    assert summary["events_with_complete_financial_values"] == 3
    assert summary["events_with_nonfinite_financial_values"] == 0

    assert summary["eps_surprise_comparisons"] == 3
    assert summary["eps_surprise_mismatches"] == 1

    assert summary["maximum_absolute_surprise_difference"] == (
        pytest.approx(0.1)
    )

    assert summary["weekend_publication_events"] == 2
    assert summary["publication_weekday_counts"]["Monday"] == 2
    assert summary["publication_weekday_counts"]["Saturday"] == 1
    assert summary["publication_weekday_counts"]["Sunday"] == 1

    pd.testing.assert_frame_equal(events, original_events)