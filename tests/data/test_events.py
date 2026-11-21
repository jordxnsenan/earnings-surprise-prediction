import pandas as pd
import pytest

from earnings_surprise.data.events import (
    build_candidate_event_ledger,
    summarize_event_quality,
    build_source_row_mapping,
    summarize_event_chronology
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
    

def test_build_source_row_mapping():
    data = make_sample_data()

    # uses a non default index to verify that source ids are based on row positions
    data.index = [10, 20, 30, 40, 50, 60]

    original_data = data.copy(deep=True)

    mapping = build_source_row_mapping(data)

    assert len(mapping) == 6

    assert mapping["source_row_id"].tolist() == [
        0, 1, 2, 3, 4, 5
    ]

    assert mapping["has_candidate_key"].tolist() == [
        True, True, True, True, True, False
    ]

    assert mapping.loc[0, "ticker"] == "A"
    assert mapping.loc[0, "date_pub"] == pd.Timestamp(
        "2024-02-01"
    )

    assert pd.isna(mapping.loc[5, "date_pub"])

    # source mapping and event ledger should agree on num. of distinct usable candidate keys
    events, _ = build_candidate_event_ledger(data)

    mapped_keys = (
        mapping.loc[
            mapping["has_candidate_key"],
            ["ticker", "date_pub"],
        ]
        .drop_duplicates()
    )

    assert len(mapped_keys) == len(events)

    pd.testing.assert_frame_equal(data, original_data)
    


def test_summarize_event_chronology():
    events = pd.DataFrame({
        "ticker": ["A", "B", "A", "A", "B", "A", "A"],
        "date_pub": pd.to_datetime([
            "2024-05-01 09:00",
            "2024-02-01",
            "2024-01-01",
            "2025-05-03",
            "2024-08-01",
            "2024-01-16",
            "2024-05-01 12:00",
        ], format="mixed"),
    })

    original_events = events.copy(deep=True)

    summary = summarize_event_chronology(events)

    assert summary["total_events"] == 7
    assert summary["unique_tickers"] == 2

    assert summary["events_per_ticker"]["minimum"] == 2
    assert summary["events_per_ticker"]["median"] == 3.5
    assert summary["events_per_ticker"]["maximum"] == 5

    assert summary["tickers_with_sufficient_history"] == 1
    assert summary["potential_four_history_windows"] == 1

    assert summary["consecutive_event_gaps"] == 5
    assert summary["gap_days"]["minimum"] == 0
    assert summary["gap_days"]["median"] == 106
    assert summary["gap_days"]["maximum"] == 367

    assert summary["same_day_event_gaps"] == 1
    assert summary["gaps_under_30_days"] == 2
    assert summary["gaps_over_180_days"] == 2
    assert summary["gaps_over_365_days"] == 1

    pd.testing.assert_frame_equal(events, original_events)


def test_chronology_rejects_duplicate_event_keys():
    events = pd.DataFrame({
        "ticker": ["A", "A"],
        "date_pub": pd.to_datetime([
            "2024-01-01",
            "2024-01-01",
        ]),
    })

    with pytest.raises(
        ValueError,
        match="Duplicate candidate earnings-event keys",
    ):
        summarize_event_chronology(events)