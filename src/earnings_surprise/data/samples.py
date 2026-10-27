import numpy as np
import pandas as pd
import json

from pathlib import Path
from earnings_surprise.data.events import REQUIRED_PRIOR_EVENTS


TARGET_COLUMNS = [
    "actual_earning_result",
    "estimated_earning",
]

EVENT_LEDGER_PATH = Path(
    "data/processed/candidate_earnings_events.parquet"
)

PREDICTION_WINDOWS_PATH = Path(
    "data/processed/prediction_windows.parquet"
)

PREDICTION_SUMMARY_PATH = Path(
    "results/prediction_samples.json"
)


def label_earnings_events(events: pd.DataFrame) -> pd.DataFrame:
    """assigns earnings-beat labels to events with valid EPS values"""
    missing_columns = [
        column for column in TARGET_COLUMNS
        if column not in events.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    labeled_events = events.copy()

    actual_eps = pd.to_numeric(
        labeled_events["actual_earning_result"],
        errors="coerce",
    )

    estimated_eps = pd.to_numeric(
        labeled_events["estimated_earning"],
        errors="coerce",
    )

    valid_target = (
        actual_eps.notna()
        & estimated_eps.notna()
        & np.isfinite(actual_eps)
        & np.isfinite(estimated_eps)
    )

    # nullable ints allow invalid targets to remain missing
    labeled_events["target_beat"] = pd.Series(
        pd.NA,
        index=labeled_events.index,
        dtype="Int8",
    )

    labeled_events.loc[valid_target, "target_beat"] = (
        actual_eps[valid_target] > estimated_eps[valid_target]
    ).astype("int8")

    return labeled_events


def build_prediction_windows(
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    
    """builds prediction samples using four preceding earnings events"""
    required_columns = [
        "ticker",
        "date_pub",
    ] + TARGET_COLUMNS

    missing_columns = [
        column for column in required_columns
        if column not in events.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    ordered_events = events[required_columns].copy()

    ordered_events["date_pub"] = pd.to_datetime(
        ordered_events["date_pub"], errors="coerce"
    )

    if (
        ordered_events["ticker"].isna().any()
        or ordered_events["date_pub"].isna().any()
    ):
        raise ValueError(
            "Event keys must contain a ticker and a valid publication date"
        )

    if ordered_events.duplicated(["ticker", "date_pub"]).any():
        raise ValueError("Duplicate candidate earnings-event keys")

    ordered_events = label_earnings_events(ordered_events)

    ordered_events = ordered_events.sort_values(
        ["ticker", "date_pub"]
    ).reset_index(drop=True)

    grouped = ordered_events.groupby("ticker", sort=False)

    history_columns = []

    previous_day = ordered_events["date_pub"].dt.normalize()

    unambiguous_history = pd.Series(
        True, index=ordered_events.index
    )

    historical_eps_missing = pd.Series(
        False, index=ordered_events.index
    )

    # links each target to its four immediately preceding events
    for lag in range(1, REQUIRED_PRIOR_EVENTS + 1):
        column = f"lag_{lag}_date_pub"

        ordered_events[column] = grouped["date_pub"].shift(lag)
        history_columns.append(column)

        history_day = ordered_events[column].dt.normalize()

        # requires a strict calendar-day order between events
        unambiguous_history &= history_day.lt(previous_day)
        previous_day = history_day

        previous_target = grouped["target_beat"].shift(lag)

        historical_eps_missing |= previous_target.isna()

    has_four_prior_events = (
        ordered_events[history_columns[-1]].notna()
    )

    ambiguous_history = (
        has_four_prior_events & ~unambiguous_history
    )

    clear_history = (
        has_four_prior_events & unambiguous_history
    )

    invalid_targets = (
        clear_history & ordered_events["target_beat"].isna()
    )

    eligible = (
        clear_history & ordered_events["target_beat"].notna()
    )

    output_columns = [
        "ticker",
        "date_pub",
        *history_columns,
        "target_beat",
    ]

    samples = (
        ordered_events.loc[eligible, output_columns]
        .rename(columns={"date_pub": "target_date_pub"})
        .reset_index(drop=True)
    )

    summary = {
        "total_candidate_events": len(ordered_events),
        "potential_four_history_windows": int(
            has_four_prior_events.sum()
        ),
        "windows_with_ambiguous_history_dates": int(
            ambiguous_history.sum()
        ),
        "windows_with_invalid_targets": int(
            invalid_targets.sum()
        ),
        "eligible_prediction_samples": len(samples),
        "eligible_samples_with_incomplete_history_eps": int(
            (eligible & historical_eps_missing).sum()
        ),
        "target_beats": int(samples["target_beat"].eq(1).sum()),
        "target_meet_or_miss": int(
            samples["target_beat"].eq(0).sum()
        ),
        "tickers_with_eligible_samples": int(
            samples["ticker"].nunique()
        ),
    }

    return samples, summary


def main() -> None:
    """generates and saves historical earnings prediction samples"""
    if not EVENT_LEDGER_PATH.is_file():
        raise FileNotFoundError(
            f"Event ledger not found: {EVENT_LEDGER_PATH}. "
            "Run the event reconstruction module first."
        )

    events = pd.read_parquet(EVENT_LEDGER_PATH)

    samples, summary = build_prediction_windows(events)

    # verifies that every potential window has been accounted for
    accounted_windows = (
        summary["windows_with_ambiguous_history_dates"]
        + summary["windows_with_invalid_targets"]
        + summary["eligible_prediction_samples"]
    )

    if accounted_windows != summary["potential_four_history_windows"]:
        raise ValueError("Prediction-window counts do not reconcile")

    PREDICTION_WINDOWS_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    PREDICTION_SUMMARY_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    samples.to_parquet(
        PREDICTION_WINDOWS_PATH,
        index=False,
    )

    PREDICTION_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"\nPrediction windows saved to: {PREDICTION_WINDOWS_PATH}")
    print(f"Summary saved to: {PREDICTION_SUMMARY_PATH}")


if __name__ == "__main__":
    main()