import pandas as pd
import pytest
import json

from earnings_surprise.data.samples import (
    EVENT_LEDGER_PATH,
    PREDICTION_WINDOWS_PATH,
    PREDICTION_SUMMARY_PATH,
    build_prediction_windows,
    label_earnings_events,
    main,
)

def test_label_earnings_events():
    events = pd.DataFrame({
        "actual_earning_result": [
            0.65, 0.40, 0.30, -0.20, None, float("inf")
        ],
        "estimated_earning": [
            0.50, 0.40, 0.45, -0.35, 0.50, 0.40
        ],
    })

    original_events = events.copy(deep=True)

    labeled = label_earnings_events(events)

    assert labeled["target_beat"].tolist()[:4] == [
        1, 0, 0, 1
    ]

    assert pd.isna(labeled.loc[4, "target_beat"])
    assert pd.isna(labeled.loc[5, "target_beat"])

    assert str(labeled["target_beat"].dtype) == "Int8"

    pd.testing.assert_frame_equal(events, original_events)


def test_label_earnings_events_requires_columns():
    events = pd.DataFrame({
        "actual_earning_result": [0.50],
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        label_earnings_events(events)    


def test_build_prediction_windows():
    events = pd.DataFrame({
        "ticker": ["A"] * 7 + ["B"] * 5 + ["C"] * 2,
        "date_pub": pd.to_datetime([
            "2024-01-01",
            "2024-02-01",
            "2024-03-01",
            "2024-04-01",
            "2024-05-01",
            "2024-06-01",
            "2024-07-01",
            "2024-01-10",
            "2024-02-10",
            "2024-03-10",
            "2024-04-10",
            "2024-05-10",
            "2024-01-12",
            "2024-02-12",
        ]),
        "actual_earning_result": [
            0.6, 0.4, 0.3, None, 0.8, None, 0.1,
            0.3, 0.3, 0.3, 0.3, 0.3,
            0.3, 0.3,
        ],
        "estimated_earning": [
            0.5, 0.5, 0.4, 0.5, 0.5, 0.5, 0.3,
            0.3, 0.3, 0.3, 0.3, 0.3,
            0.2, 0.2,
        ],
    })

    # shuffles the events to ensure the function sorts them
    events = events.sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    original_events = events.copy(deep=True)

    samples, summary = build_prediction_windows(events)

    assert summary["total_candidate_events"] == 14
    assert summary["potential_four_history_windows"] == 4
    assert summary["windows_with_ambiguous_history_dates"] == 0
    assert summary["windows_with_invalid_targets"] == 1

    assert summary["eligible_prediction_samples"] == 3
    assert summary["eligible_samples_with_incomplete_history_eps"] == 2

    assert summary["target_beats"] == 1
    assert summary["target_meet_or_miss"] == 2

    assert summary["tickers_with_eligible_samples"] == 2

    # verifies the four preceding events for A's May target
    may_sample = samples.loc[
        (samples["ticker"] == "A")
        & (samples["target_date_pub"] == pd.Timestamp("2024-05-01"))
    ].iloc[0]

    assert may_sample["lag_1_date_pub"] == pd.Timestamp("2024-04-01")
    assert may_sample["lag_4_date_pub"] == pd.Timestamp("2024-01-01")
    assert may_sample["target_beat"] == 1

    # junes has an invalid target but must remain in Julys history
    july_sample = samples.loc[
        (samples["ticker"] == "A")
        & (samples["target_date_pub"] == pd.Timestamp("2024-07-01"))
    ].iloc[0]

    assert july_sample["lag_1_date_pub"] == pd.Timestamp("2024-06-01")
    assert july_sample["target_beat"] == 0

    # historical windows must never cross ticker boundaries
    b_sample = samples.loc[samples["ticker"] == "B"].iloc[0]

    assert b_sample["lag_4_date_pub"] == pd.Timestamp("2024-01-10")

    pd.testing.assert_frame_equal(events, original_events)


def test_prediction_windows_reject_ambiguous_same_day_events():
    events = pd.DataFrame({
        "ticker": ["A"] * 6,
        "date_pub": [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-02-01 09:00"),
            pd.Timestamp("2024-02-01 12:00"),
            pd.Timestamp("2024-03-01"),
            pd.Timestamp("2024-04-01"),
            pd.Timestamp("2024-05-01"),
        ],
        "actual_earning_result": [0.5] * 6,
        "estimated_earning": [0.4] * 6,
    })

    samples, summary = build_prediction_windows(events)

    assert summary["potential_four_history_windows"] == 2
    assert summary["windows_with_ambiguous_history_dates"] == 2
    assert summary["eligible_prediction_samples"] == 0

    assert samples.empty


def test_prediction_windows_reject_duplicate_event_keys():
    events = pd.DataFrame({
        "ticker": ["A", "A"],
        "date_pub": pd.to_datetime([
            "2024-01-01",
            "2024-01-01",
        ]),
        "actual_earning_result": [0.5, 0.5],
        "estimated_earning": [0.4, 0.4],
    })

    with pytest.raises(
        ValueError,
        match="Duplicate candidate earnings-event keys",
    ):
        build_prediction_windows(events)
        

def test_main_generates_prediction_artifacts(tmp_path, monkeypatch):
    events = pd.DataFrame({
        "ticker": ["A"] * 5,
        "date_pub": pd.date_range(
            "2024-01-01", periods=5, freq="MS"
        ),
        "actual_earning_result": [
            0.3, 0.4, 0.2, 0.5, 0.7
        ],
        "estimated_earning": [
            0.4, 0.4, 0.4, 0.4, 0.4
        ],
    })

    input_path = tmp_path / EVENT_LEDGER_PATH
    input_path.parent.mkdir(parents=True)

    events.to_parquet(input_path, index=False)

    # runs the command from an isolated temporary directory
    monkeypatch.chdir(tmp_path)

    main()

    output_path = tmp_path / PREDICTION_WINDOWS_PATH
    summary_path = tmp_path / PREDICTION_SUMMARY_PATH

    assert output_path.is_file()
    assert summary_path.is_file()

    saved_samples = pd.read_parquet(output_path)

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    assert summary["total_candidate_events"] == 5
    assert summary["potential_four_history_windows"] == 1
    assert summary["eligible_prediction_samples"] == 1
    assert summary["target_beats"] == 1

    assert len(saved_samples) == 1
    assert saved_samples.loc[0, "target_beat"] == 1

    assert saved_samples.loc[
        0, "lag_4_date_pub"
    ] == pd.Timestamp("2024-01-01")

    # targets financial values must not enter the feature links
    assert "actual_earning_result" not in saved_samples.columns
    assert "estimated_earning" not in saved_samples.columns