
import json
from pathlib import Path

import pandas as pd

from earnings_surprise.data.download import (
    DATASET_FILENAME,
    DATASET_REPOSITORY,
    DATASET_REVISION,
    EXPECTED_SHA256,
    calculate_sha256,
)


def summarize_date_column(values: pd.Series) -> dict:
    """summarizes the coverage + quality of a date column"""
    parsed_dates = pd.to_datetime(values, errors="coerce")
    valid_dates = parsed_dates.dropna()

    first_date = None
    last_date = None

    if not valid_dates.empty:
        first_date = valid_dates.min().date().isoformat()
        last_date = valid_dates.max().date().isoformat()

    rows_by_year = {
        str(int(year)): int(count)
        for year, count in parsed_dates.dt.year.value_counts().sort_index().items()
    }

    return {
        "first": first_date,
        "last": last_date,
        "missing_values": int(values.isna().sum()),
        "unparseable_values": int(
            (values.notna() & parsed_dates.isna()).sum()
        ),
        "rows_by_year": rows_by_year,
    }


def summarize_dataset(data: pd.DataFrame) -> dict:
    """generates summary statistics for the raw earnings data"""
    required_columns = ["ticker", "date", "date_pub"]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    publication_dates = pd.to_datetime(
        data["date_pub"], errors="coerce"
    )

    candidate_keys = pd.DataFrame({
        "ticker": data["ticker"],
        "date_pub": publication_dates,
    }).dropna()

    unique_candidate_keys = len(candidate_keys.drop_duplicates())

    return {
        "raw_rows": len(data),
        "column_count": len(data.columns),
        "column_types": {
            column: str(data[column].dtype)
            for column in data.columns
        },
        "missing_values": {
            column: int(data[column].isna().sum())
            for column in data.columns
        },
        "unique_tickers": int(
            data["ticker"].nunique(dropna=True)
        ),
        "date_fields": {
            "date": summarize_date_column(data["date"]),
            "date_pub": summarize_date_column(data["date_pub"]),
        },
        "rows_with_candidate_key": len(candidate_keys),
        "rows_without_candidate_key": len(data) - len(candidate_keys),
        "candidate_ticker_publication_pairs": unique_candidate_keys,
        "repeated_candidate_key_rows": (
            len(candidate_keys) - unique_candidate_keys
        ),
    }


def main() -> None:
    dataset_path = Path("data/raw") / DATASET_FILENAME
    output_path = Path("results/data_inspection.json")

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Run the dataset downloader first."
        )

    actual_checksum = calculate_sha256(dataset_path)

    if actual_checksum != EXPECTED_SHA256:
        raise ValueError("Dataset checksum does not match the source.")

    data = pd.read_parquet(dataset_path)
    summary = summarize_dataset(data)

    summary["source"] = {
        "repository": DATASET_REPOSITORY,
        "revision": DATASET_REVISION,
        "filename": DATASET_FILENAME,
        "sha256": actual_checksum,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Raw records: {summary['raw_rows']:,}")
    print(f"Columns: {summary['column_count']}")
    print(f"Unique tickers: {summary['unique_tickers']:,}")
    
    print(
        "Candidate ticker/publication pairs: "
        f"{summary['candidate_ticker_publication_pairs']:,}"
    )
    
    print(
        "Repeated candidate-key rows: "
        f"{summary['repeated_candidate_key_rows']:,}"
    )
    
    print(
        "Rows without a usable candidate key: "
        f"{summary['rows_without_candidate_key']:,}"
    )
    
    print(f"Inspection saved to: {output_path}")


if __name__ == "__main__":
    main()