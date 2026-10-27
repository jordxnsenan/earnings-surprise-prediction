import json
from pathlib import Path

import pandas as pd
import numpy as np

from earnings_surprise.data.download import (
    DATASET_FILENAME,
    EXPECTED_SHA256,
    calculate_sha256,
)


CANDIDATE_KEY_COLUMNS = ["ticker", "date_pub"]

FINANCIAL_COLUMNS = [
    "actual_earning_result",
    "estimated_earning",
    "eps_surprise",
]

REQUIRED_COLUMNS = (
    CANDIDATE_KEY_COLUMNS + ["date"] + FINANCIAL_COLUMNS
)

EPS_SURPRISE_TOLERANCE = 1e-6

REQUIRED_PRIOR_EVENTS = 4
SHORT_EVENT_GAP_DAYS = 30
LONG_EVENT_GAP_DAYS = 180
VERY_LONG_EVENT_GAP_DAYS = 365

def build_candidate_event_ledger(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """consolidates consistent snapshots into candidate earnings events"""
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    for column in FINANCIAL_COLUMNS:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise TypeError(f"Expected numeric values in {column}")

    publication_dates = pd.to_datetime(
        data["date_pub"], errors="coerce"
    )

    source_dates = pd.to_datetime(
        data["date"], errors="coerce"
    )

    valid_key_mask = (
        data["ticker"].notna() & publication_dates.notna()
    )

    records = data.loc[valid_key_mask, REQUIRED_COLUMNS].copy()

    records["date_pub"] = publication_dates.loc[valid_key_mask]
    records["date"] = source_dates.loc[valid_key_mask]

    grouped = records.groupby(
        CANDIDATE_KEY_COLUMNS,
        sort=True,
    )

    # confirms that each candidate event has at most one distinct non missing value per financial field
    minimum_values = grouped[FINANCIAL_COLUMNS].min()
    maximum_values = grouped[FINANCIAL_COLUMNS].max()

    has_conflicts = (
        minimum_values.notna()
        & maximum_values.notna()
        & minimum_values.ne(maximum_values)
    )

    if has_conflicts.any().any():
        conflict_counts = has_conflicts.sum().to_dict()

        raise ValueError(
            "Conflicting non-missing financial values: "
            f"{conflict_counts}"
        )

    source_counts = grouped.size().rename("source_row_count")

    source_date_ranges = (
        grouped["date"]
        .agg(["min", "max"])
        .rename(columns={
            "min": "first_source_date",
            "max": "last_source_date",
        })
    )

    observed_counts = (
        grouped[FINANCIAL_COLUMNS]
        .count()
        .add_suffix("_observed_rows")
    )

    # minima are safe here because conflicting non missing vals. have already been rejected
    event_ledger = pd.concat(
        [
            minimum_values,
            source_counts,
            source_date_ranges,
            observed_counts,
        ],
        axis=1,
    ).reset_index()

    summary = {
        "raw_rows": len(data),
        "rows_with_candidate_key": len(records),
        "rows_without_candidate_key": int(
            (~valid_key_mask).sum()
        ),
        "candidate_event_count": len(event_ledger),
        "events_with_multiple_source_rows": int(
            (event_ledger["source_row_count"] > 1).sum()
        ),
        "events_missing_actual_eps": int(
            event_ledger["actual_earning_result"].isna().sum()
        ),
        "events_missing_estimated_eps": int(
            event_ledger["estimated_earning"].isna().sum()
        ),
        "events_missing_supplied_surprise": int(
            event_ledger["eps_surprise"].isna().sum()
        ),
    }

    return event_ledger, summary


def summarize_event_quality(events: pd.DataFrame) -> dict:
    """validates EPS surprise calculations and event publication dates"""
    required_columns = CANDIDATE_KEY_COLUMNS + FINANCIAL_COLUMNS

    missing_columns = [
        column for column in required_columns
        if column not in events.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    financial_values = events[FINANCIAL_COLUMNS]

    complete_financial_values = financial_values.notna().all(axis=1)

    finite_financial_values = pd.Series(
        np.isfinite(
            financial_values.to_numpy(dtype=float)
        ).all(axis=1),
        index=events.index,
    )

    valid_financial_values = (
        complete_financial_values & finite_financial_values
    )

    comparable_events = events.loc[valid_financial_values]

    calculated_surprises = (
        comparable_events["actual_earning_result"]
        - comparable_events["estimated_earning"]
    )

    surprise_differences = (
        comparable_events["eps_surprise"] - calculated_surprises
    ).abs()

    inconsistent_surprises = (
        surprise_differences > EPS_SURPRISE_TOLERANCE
    )

    maximum_difference = None

    if not surprise_differences.empty:
        maximum_difference = float(surprise_differences.max())

    publication_dates = pd.to_datetime(
        events["date_pub"], errors="coerce"
    )

    publication_weekdays = publication_dates.dt.day_name()

    weekday_counts = {
        day: int((publication_weekdays == day).sum())
        for day in [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
    }

    weekend_publications = publication_weekdays.isin(
        ["Saturday", "Sunday"]
    )

    return {
        "total_events": len(events),
        "events_with_complete_financial_values": int(
            complete_financial_values.sum()
        ),
        "events_with_nonfinite_financial_values": int(
            (complete_financial_values & ~finite_financial_values).sum()
        ),
        "eps_surprise_comparisons": len(comparable_events),
        "eps_surprise_mismatches": int(
            inconsistent_surprises.sum()
        ),
        "maximum_absolute_surprise_difference": maximum_difference,
        "publication_weekday_counts": weekday_counts,
        "weekend_publication_events": int(
            weekend_publications.sum()
        ),
        "events_with_invalid_publication_dates": int(
            publication_dates.isna().sum()
        ),
    }


def build_source_row_mapping(data: pd.DataFrame) -> pd.DataFrame:
    """maps each source row to its candidate earnings event key"""
    required_columns = ["ticker", "date", "date_pub"]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    mapping = data[required_columns].reset_index(drop=True).copy()

    # preserves original row positions in the source dataset
    mapping.insert(0, "source_row_id", range(len(mapping)))

    mapping["date_pub"] = pd.to_datetime(
        mapping["date_pub"], errors="coerce"
    )

    mapping["has_candidate_key"] = (
        mapping["ticker"].notna()
        & mapping["date_pub"].notna()
    )

    return mapping


def summarize_event_chronology(events: pd.DataFrame) -> dict:
    """analyzes event counts and publication gaps by ticker"""
    required_columns = ["ticker", "date_pub"]

    missing_columns = [
        column for column in required_columns
        if column not in events.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if events.empty:
        raise ValueError("No earnings events available")

    chronology = events[required_columns].copy()

    chronology["date_pub"] = pd.to_datetime(
        chronology["date_pub"], errors="coerce"
    )

    if chronology.isna().any().any():
        raise ValueError("Invalid event identifiers or publication dates")

    if chronology.duplicated(subset=required_columns).any():
        raise ValueError("Duplicate candidate earnings-event keys")

    # sorts events chronologically within each company
    chronology = chronology.sort_values(
        ["ticker", "date_pub"]
    ).reset_index(drop=True)

    chronology["publication_day"] = (
        chronology["date_pub"].dt.normalize()
    )

    event_counts = chronology.groupby("ticker").size()

    # calculates calendar day gaps between consec. events
    gap_days = (
        chronology.groupby("ticker")["publication_day"]
        .diff()
        .dt.days
        .dropna()
    )

    # finds num. of possible targets with four preceding recorded events
    potential_windows = (
        (event_counts - REQUIRED_PRIOR_EVENTS)
        .clip(lower=0)
        .sum()
    )

    return {
        "total_events": len(chronology),
        "unique_tickers": len(event_counts),
        "events_per_ticker": {
            "minimum": int(event_counts.min()),
            "median": float(event_counts.median()),
            "maximum": int(event_counts.max()),
        },
        "tickers_with_sufficient_history": int(
            (event_counts > REQUIRED_PRIOR_EVENTS).sum()
        ),
        "potential_four_history_windows": int(potential_windows),
        "consecutive_event_gaps": len(gap_days),
        "gap_days": {
            "minimum": (
                int(gap_days.min()) if not gap_days.empty else None
            ),
            "median": (
                float(gap_days.median()) if not gap_days.empty else None
            ),
            "maximum": (
                int(gap_days.max()) if not gap_days.empty else None
            ),
        },
        "same_day_event_gaps": int((gap_days == 0).sum()),
        "gaps_under_30_days": int(
            (gap_days < SHORT_EVENT_GAP_DAYS).sum()
        ),
        "gaps_over_180_days": int(
            (gap_days > LONG_EVENT_GAP_DAYS).sum()
        ),
        "gaps_over_365_days": int(
            (gap_days > VERY_LONG_EVENT_GAP_DAYS).sum()
        ),
    }


def main() -> None:
    dataset_path = Path("data/raw") / DATASET_FILENAME
    
    output_path = Path("data/processed/candidate_earnings_events.parquet")
    summary_path = Path("results/event_reconstruction.json")
    source_mapping_path = Path("data/processed/event_source_rows.parquet")

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    if calculate_sha256(dataset_path) != EXPECTED_SHA256:
        raise ValueError("Source dataset checksum mismatch")

    data = pd.read_parquet(dataset_path)

    event_ledger, summary = build_candidate_event_ledger(data)
    summary["event_quality"] = summarize_event_quality(event_ledger)
    summary["event_chronology"] = summarize_event_chronology(event_ledger)
    
    source_mapping = build_source_row_mapping(data)

    summary["source_lineage"] = {
        "total_source_rows": len(source_mapping),
        "rows_with_candidate_keys": int(
            source_mapping["has_candidate_key"].sum()
        ),
        "rows_without_candidate_keys": int(
            (~source_mapping["has_candidate_key"]).sum()
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    event_ledger.to_parquet(output_path, index=False)
    
    source_mapping.to_parquet(
        source_mapping_path,
        index=False,
    )

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"\nEvent ledger saved to: {output_path}")
    print(f"Source mapping saved to: {source_mapping_path}")


if __name__ == "__main__":
    main()