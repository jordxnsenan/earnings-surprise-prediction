
import json
import pandas as pd

from pathlib import Path
from earnings_surprise.data.download import DATASET_FILENAME


CANDIDATE_KEY_COLUMNS = ["ticker", "date_pub"]

FINANCIAL_COLUMNS = [
    "actual_earning_result",
    "estimated_earning",
    "eps_surprise",
]

DATE_EXAMPLE_COLUMNS = [
    "ticker",
    "date",
    "date_pub",
    "actual_earning_result",
    "estimated_earning",
    "eps_surprise",
]

DEFAULT_EXAMPLE_COUNT = 5


def summarize_snapshot_repetition(data: pd.DataFrame) -> dict:
    """analyzes repeated records + conflicting financial values"""
    required_columns = CANDIDATE_KEY_COLUMNS + FINANCIAL_COLUMNS

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # only records w/ usable candidate keys can be grouped
    valid_key_mask = data[CANDIDATE_KEY_COLUMNS].notna().all(axis=1)
    valid_records = data.loc[valid_key_mask]

    # groups records by ticker + publication date
    grouped = valid_records.groupby(
        CANDIDATE_KEY_COLUMNS,
        sort=False,
    )

    rows_per_key = grouped.size()

    # counts distinct non missing vals. within each candidate event
    distinct_values = grouped[FINANCIAL_COLUMNS].nunique(
        dropna=True
    )

    conflicting_values = {}

    for column in FINANCIAL_COLUMNS:
        conflicting_values[column] = int(
            (distinct_values[column] > 1).sum()
        )

    repeated_keys = rows_per_key[rows_per_key > 1]

    max_rows_per_key = 0

    if not rows_per_key.empty:
        max_rows_per_key = int(rows_per_key.max())

    return {
        "total_rows": len(data),
        "rows_with_candidate_key": len(valid_records),
        "rows_without_candidate_key": (
            len(data) - len(valid_records)
        ),
        "candidate_key_count": len(rows_per_key),
        "keys_with_multiple_rows": len(repeated_keys),
        "rows_in_repeated_keys": int(repeated_keys.sum()),
        "extra_rows_from_repetition": int(
            (rows_per_key - 1).sum()
        ),
        "exact_duplicate_rows": int(data.duplicated().sum()),
        "maximum_rows_per_key": max_rows_per_key,
        "keys_with_conflicting_financial_values": conflicting_values,
    }


def summarize_date_relationships(data: pd.DataFrame) -> dict:
    """audits snapshot dates relative to earnings publication dates"""
    
    required_columns = [
        "ticker",
        "date",
        "date_pub",
        "actual_earning_result",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    snapshot_dates = pd.to_datetime(
        data["date"], errors="coerce"
    ).dt.normalize()

    publication_dates = pd.to_datetime(
        data["date_pub"], errors="coerce"
    ).dt.normalize()

    valid_dates = snapshot_dates.notna() & publication_dates.notna()

    before_publication = valid_dates & (
        snapshot_dates < publication_dates
    )

    on_publication = valid_dates & (
        snapshot_dates == publication_dates
    )

    after_publication = valid_dates & (
        snapshot_dates > publication_dates
    )

    has_actual_eps = pd.to_numeric(
        data["actual_earning_result"], errors="coerce"
    ).notna()

    # identifies candidate events q/ recorded actual EPS appearing in rows dated before their pub. date
    prepublication_events = data.loc[
        before_publication & has_actual_eps,
        ["ticker", "date_pub"],
    ].dropna().drop_duplicates()

    day_gaps = (
        publication_dates[valid_dates] -
        snapshot_dates[valid_dates]
    ).dt.days

    return {
        "rows_with_both_dates": int(valid_dates.sum()),
        "rows_missing_or_invalid_either_date": int(
            (~valid_dates).sum()
        ),
        "rows_before_publication": int(before_publication.sum()),
        "rows_on_publication_date": int(on_publication.sum()),
        "rows_after_publication": int(after_publication.sum()),
        "prepublication_rows_with_recorded_actual_eps": int(
            (before_publication & has_actual_eps).sum()
        ),
        "candidate_events_with_prepublication_actual_eps": int(
            len(prepublication_events)
        ),
        "publication_minus_snapshot_days": {
            "minimum": (
                int(day_gaps.min()) if not day_gaps.empty else None
            ),
            "median": (
                float(day_gaps.median()) if not day_gaps.empty else None
            ),
            "maximum": (
                int(day_gaps.max()) if not day_gaps.empty else None
            ),
        },
    }
    

def find_date_examples(
    data: pd.DataFrame,
    example_count: int = DEFAULT_EXAMPLE_COUNT,
) -> dict:
    """Find examples of unusual snapshot and publication dates."""
    missing_columns = [
        column for column in DATE_EXAMPLE_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if example_count < 1:
        raise ValueError("example_count must be positive")

    records = data[DATE_EXAMPLE_COLUMNS].copy()

    records["date"] = pd.to_datetime(
        records["date"], errors="coerce"
    ).dt.normalize()

    records["date_pub"] = pd.to_datetime(
        records["date_pub"], errors="coerce"
    ).dt.normalize()

    records["actual_earning_result"] = pd.to_numeric(
        records["actual_earning_result"], errors="coerce"
    )

    records = records.dropna(
        subset=["ticker", "date", "date_pub"]
    )

    records["days_before_publication"] = (
        records["date_pub"] - records["date"]
    ).dt.days

    has_early_actual = (
        (records["days_before_publication"] > 0)
        & records["actual_earning_result"].notna()
    )

    longest_gaps = (
        records.loc[has_early_actual]
        .sort_values("days_before_publication", ascending=False)
        .drop_duplicates(subset=CANDIDATE_KEY_COLUMNS)
        .head(example_count)
    )

    postpublication_examples = (
        records.loc[records["days_before_publication"] < 0]
        .sort_values("days_before_publication")
        .drop_duplicates(subset=CANDIDATE_KEY_COLUMNS)
        .head(example_count)
    )

    return {
        "largest_prepublication_gaps": longest_gaps,
        "postpublication_examples": postpublication_examples,
    }


def main() -> None:
    dataset_path = Path("data/raw") / DATASET_FILENAME
    output_path = Path("results/snapshot_audit.json")

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Run the dataset downloader first."
        )

    data = pd.read_parquet(dataset_path)
    summary = summarize_snapshot_repetition(data)
    summary["date_relationships"] = summarize_date_relationships(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"\nAudit saved to: {output_path}")
    
    examples = find_date_examples(data)

    for category, example_records in examples.items():
        print(f"\n{category}:")

        if example_records.empty:
            print("No examples found.")
        
        else:
            print(example_records.to_string(index=False))


if __name__ == "__main__":
    main()