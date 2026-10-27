
import pandas as pd
import pytest

from earnings_surprise.data.inspect import (
    summarize_date_column,
    summarize_dataset,
)


def test_summarize_dataset():
    data = pd.DataFrame({
        "ticker": ["A", "A", "A", "B", None],
        "date": pd.to_datetime([
            "2017-01-06",
            "2017-01-13",
            "2017-04-28",
            "2017-01-20",
            "2017-01-27",
        ]),
        "date_pub": pd.to_datetime([
            "2017-02-14",
            "2017-02-14",
            "2017-05-22",
            "2017-02-14",
            "2017-05-22",
        ]),
        "actual_earning_result": [
            0.53, 0.53, 0.58, None, 0.33
        ],
    })

    summary = summarize_dataset(data)

    assert summary["raw_rows"] == 5
    assert summary["column_count"] == 4
    assert summary["unique_tickers"] == 2

    assert summary["candidate_ticker_publication_pairs"] == 3
    assert summary["repeated_candidate_key_rows"] == 1
    assert summary["rows_with_candidate_key"] == 4
    assert summary["rows_without_candidate_key"] == 1

    assert summary["missing_values"]["ticker"] == 1
    assert summary["missing_values"]["actual_earning_result"] == 1


def test_summarize_date_column():
    dates = pd.Series([
        "2024-01-01",
        "invalid-date",
        None,
    ])

    summary = summarize_date_column(dates)

    assert summary["first"] == "2024-01-01"
    assert summary["last"] == "2024-01-01"
    assert summary["missing_values"] == 1
    assert summary["unparseable_values"] == 1


def test_summarize_dataset_requires_date_columns():
    data = pd.DataFrame({
        "ticker": ["A"],
        "date": ["2024-01-01"],
    })

    with pytest.raises(ValueError, match="date_pub"):
        summarize_dataset(data)
