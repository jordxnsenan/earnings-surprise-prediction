
import numpy as np
import pandas as pd
import json

from pathlib import Path
from earnings_surprise.data.events import (
    CANDIDATE_KEY_COLUMNS,
    REQUIRED_PRIOR_EVENTS,
)


HISTORICAL_VALUE_COLUMNS = [
    "actual_earning_result",
    "estimated_earning",
]

TREND_POSITION_WEIGHTS = (-1.5, -0.5, 0.5, 1.5)

TREND_DENOMINATOR = sum(
    weight ** 2 for weight in TREND_POSITION_WEIGHTS
)

DERIVED_FEATURE_COLUMNS = [
    "beat_rate4",
    "surprise_mean4",
    "surprise_std4",
    "eps_change1_2",
    "eps_trend4",
    "surprise_trend4",
    "beat_streak4",
    "previous_report_month",
    "previous_report_quarter",
]

FEATURE_COLUMNS = (
    [f"actual_eps_lag{lag}" for lag in range(1, REQUIRED_PRIOR_EVENTS + 1)]
    + [f"estimate_eps_lag{lag}" for lag in range(1, REQUIRED_PRIOR_EVENTS + 1)]
    + [f"surprise_eps_lag{lag}" for lag in range(1, REQUIRED_PRIOR_EVENTS + 1)]
    + [f"beat_lag{lag}" for lag in range(1, REQUIRED_PRIOR_EVENTS + 1)]
    + DERIVED_FEATURE_COLUMNS
)

EVENT_LEDGER_PATH = Path(
    "data/processed/candidate_earnings_events.parquet"
)

PREDICTION_WINDOWS_PATH = Path(
    "data/processed/prediction_windows.parquet"
)

FEATURE_MATRIX_PATH = Path(
    "data/processed/historical_features.parquet"
)

FEATURE_SUMMARY_PATH = Path(
    "results/feature_summary.json"
)


def build_lagged_features(
    samples: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """generates earnings features from four preceding events"""
    lag_date_columns = [
        f"lag_{lag}_date_pub"
        for lag in range(1, REQUIRED_PRIOR_EVENTS + 1)
    ]

    required_sample_columns = [
        "ticker",
        "target_date_pub",
        *lag_date_columns,
    ]

    required_event_columns = (
        CANDIDATE_KEY_COLUMNS + HISTORICAL_VALUE_COLUMNS
    )

    for name, data, required_columns in [
        ("sample", samples, required_sample_columns),
        ("event", events, required_event_columns),
    ]:
        missing_columns = [
            column for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing {name} columns: {missing_columns}"
            )

    event_records = events[required_event_columns].copy()

    event_records["date_pub"] = pd.to_datetime(
        event_records["date_pub"], errors="raise"
    )

    if event_records[CANDIDATE_KEY_COLUMNS].isna().any().any():
        raise ValueError("Event keys cannot be missing")

    if event_records.duplicated(CANDIDATE_KEY_COLUMNS).any():
        raise ValueError("Duplicate earnings-event keys")

    if samples[["ticker", "target_date_pub"]].isna().any().any():
        raise ValueError("Sample identifiers cannot be missing")

    if samples.duplicated(["ticker", "target_date_pub"]).any():
        raise ValueError("Duplicate prediction samples")

    event_values = event_records.set_index(
        CANDIDATE_KEY_COLUMNS
    )[HISTORICAL_VALUE_COLUMNS]

    previous_dates = pd.to_datetime(
        samples["target_date_pub"], errors="raise"
    ).dt.normalize()

    features = pd.DataFrame(index=samples.index)

    for lag in range(1, REQUIRED_PRIOR_EVENTS + 1):
        lag_column = f"lag_{lag}_date_pub"

        history_dates = pd.to_datetime(
            samples[lag_column], errors="raise"
        )

        history_days = history_dates.dt.normalize()

        if not history_days.lt(previous_dates).all():
            raise ValueError(
                "Historical events must occur on strictly earlier "
                "calendar days"
            )

        previous_dates = history_days

        # matches the historical event using both its ticker and date
        reference_keys = pd.MultiIndex.from_arrays(
            [samples["ticker"], history_dates],
            names=CANDIDATE_KEY_COLUMNS,
        )

        if not reference_keys.isin(event_values.index).all():
            raise ValueError(
                f"Lag {lag} references an unknown earnings event"
            )

        historical_records = event_values.reindex(reference_keys)

        actual = pd.Series(
            pd.to_numeric(
                historical_records["actual_earning_result"],
                errors="coerce",
            ).to_numpy(),
            index=samples.index,
            dtype="float64",
        )

        estimate = pd.Series(
            pd.to_numeric(
                historical_records["estimated_earning"],
                errors="coerce",
            ).to_numpy(),
            index=samples.index,
            dtype="float64",
        )

        # preserves missing or non-finite historical values
        actual = actual.where(np.isfinite(actual))
        estimate = estimate.where(np.isfinite(estimate))

        features[f"actual_eps_lag{lag}"] = actual
        features[f"estimate_eps_lag{lag}"] = estimate
        features[f"surprise_eps_lag{lag}"] = actual - estimate

        beat = pd.Series(
            pd.NA,
            index=samples.index,
            dtype="Int8",
        )

        has_both = actual.notna() & estimate.notna()

        beat.loc[has_both] = (
            actual.loc[has_both] > estimate.loc[has_both]
        ).astype("int8")

        features[f"beat_lag{lag}"] = beat

    return features


def calculate_four_event_trend(values: pd.DataFrame) -> pd.Series:
    """calculates the linear trend across four recorded events"""
    weighted_values = values.mul(
        TREND_POSITION_WEIGHTS,
        axis=1,
    )

    return (
        weighted_values.sum(
            axis=1,
            min_count=REQUIRED_PRIOR_EVENTS,
        )
        / TREND_DENOMINATOR
    )
    

def build_derived_features(
    samples: pd.DataFrame,
    lagged_features: pd.DataFrame,
) -> pd.DataFrame:
    """calculates summary and calendar features from historical events"""
    
    actual_columns = [
        f"actual_eps_lag{lag}"
        for lag in range(1, REQUIRED_PRIOR_EVENTS + 1)
    ]

    surprise_columns = [
        f"surprise_eps_lag{lag}"
        for lag in range(1, REQUIRED_PRIOR_EVENTS + 1)
    ]

    beat_columns = [
        f"beat_lag{lag}"
        for lag in range(1, REQUIRED_PRIOR_EVENTS + 1)
    ]

    required_columns = (
        actual_columns + surprise_columns + beat_columns
    )

    missing_columns = [
        column for column in required_columns
        if column not in lagged_features.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing lagged feature columns: {missing_columns}"
        )

    if "lag_1_date_pub" not in samples.columns:
        raise ValueError("Missing sample column: lag_1_date_pub")

    if not samples.index.equals(lagged_features.index):
        raise ValueError("Sample and feature indices must match")

    # converts numeric inputs and preserve invalid values as missing
    values = (
        lagged_features[required_columns]
        .apply(pd.to_numeric, errors="coerce")
        .astype("float64")
        .replace([np.inf, -np.inf], np.nan)
    )

    actual = values[actual_columns]
    surprise = values[surprise_columns]
    beats = values[beat_columns]

    invalid_beats = beats.notna() & ~beats.isin([0, 1])

    if invalid_beats.any().any():
        raise ValueError("Beat indicators must be 0, 1, or missing")

    features = pd.DataFrame(index=samples.index)

    # historical beat rate and surprise statistics
    features["beat_rate4"] = beats.mean(
        axis=1, skipna=False
    )

    features["surprise_mean4"] = surprise.mean(
        axis=1, skipna=False
    )

    features["surprise_std4"] = surprise.std(
        axis=1, ddof=0, skipna=False
    )

    # changes between the two most recent actual EPS values
    features["eps_change1_2"] = (
        actual["actual_eps_lag1"]
        - actual["actual_eps_lag2"]
    )

    # calculates trends in oldest to newest order
    features["eps_trend4"] = calculate_four_event_trend(
        actual[actual_columns[::-1]]
    )

    features["surprise_trend4"] = calculate_four_event_trend(
        surprise[surprise_columns[::-1]]
    )

    # counts consec. beats beginning at lag 1
    consecutive_beats = (
        beats.fillna(0)
        .eq(1)
        .cumprod(axis=1)
        .sum(axis=1)
    )

    features["beat_streak4"] = (
        consecutive_beats.astype("Int8")
        .where(beats.notna().all(axis=1), pd.NA)
    )

    # calendar feat. come from the preceding event
    previous_dates = pd.to_datetime(
        samples["lag_1_date_pub"],
        errors="coerce",
    )

    if previous_dates.isna().any():
        raise ValueError("Invalid previous report dates")

    months = previous_dates.dt.month
    quarters = (months - 1) // 3 + 1

    features["previous_report_month"] = pd.Categorical(
        months,
        categories=range(1, 13),
    )

    features["previous_report_quarter"] = pd.Categorical(
        quarters,
        categories=range(1, 5),
    )

    return features


def build_historical_features(
    samples: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """generates the complete 25 feature historical input matrix"""
    
    lagged_features = build_lagged_features(samples, events)

    derived_features = build_derived_features(
        samples,
        lagged_features,
    )

    features = pd.concat(
        [lagged_features, derived_features],
        axis=1,
    )

    return features[FEATURE_COLUMNS]


def main() -> None:
    """generates and saves the complete historical feature matrix"""
   
    required_files = [
        EVENT_LEDGER_PATH,
        PREDICTION_WINDOWS_PATH,
    ]

    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required dataset not found: {file_path}"
            )

    events = pd.read_parquet(EVENT_LEDGER_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    features = build_historical_features(samples, events)

    # verifies that no samples or predictors were lost
    if len(features) != len(samples):
        raise ValueError("Feature and sample counts do not match")

    if features.columns.tolist() != FEATURE_COLUMNS:
        raise ValueError("Feature columns do not match the specification")

    numeric_features = features.select_dtypes(include="number")

    numeric_values = numeric_features.to_numpy(
        dtype="float64",
        na_value=np.nan,
    )

    nonfinite_count = int(np.isinf(numeric_values).sum())

    if nonfinite_count > 0:
        raise ValueError(
            f"Found {nonfinite_count} infinite feature values"
        )

    missing_counts = features.isna().sum()

    summary = {
        "prediction_samples": len(features),
        "feature_count": len(features.columns),
        "numerical_features": len(numeric_features.columns),
        "categorical_features": len(
            features.select_dtypes(include="category").columns
        ),
        "rows_with_missing_features": int(
            features.isna().any(axis=1).sum()
        ),
        "missing_values_by_feature": {
            column: int(missing_counts[column])
            for column in FEATURE_COLUMNS
        },
        "feature_types": {
            column: str(features[column].dtype)
            for column in FEATURE_COLUMNS
        },
    }

    FEATURE_MATRIX_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    FEATURE_SUMMARY_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    # feature rows preserve the order of prediction_windows.parquet
    features.to_parquet(FEATURE_MATRIX_PATH, index=False)

    FEATURE_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"\nFeature matrix saved to: {FEATURE_MATRIX_PATH}")
    print(f"Summary saved to: {FEATURE_SUMMARY_PATH}")


if __name__ == "__main__":
    main()