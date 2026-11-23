import pandas as pd
import pytest
import json

from earnings_surprise.features import (
    build_derived_features,
    build_historical_features,
    build_lagged_features,
    EVENT_LEDGER_PATH,
    FEATURE_COLUMNS,
    FEATURE_MATRIX_PATH,
    FEATURE_SUMMARY_PATH,
    PREDICTION_WINDOWS_PATH,
    main,
)

def make_feature_test_data():
    events = pd.DataFrame({
        "ticker": ["A", "A", "A", "A", "A", "B"],
        "date_pub": pd.to_datetime([
            "2022-01-25",
            "2022-04-26",
            "2022-07-26",
            "2022-10-25",
            "2023-01-24",
            "2022-10-25",
        ]),
        "actual_earning_result": [
            1.00, -0.50, 0.60, 0.90, 99.0, 10.0
        ],
        "estimated_earning": [
            0.80, -0.40, 0.60, 0.70, -99.0, 5.0
        ],
    })

    samples = pd.DataFrame({
        "ticker": ["A"],
        "target_date_pub": pd.to_datetime(["2023-01-24"]),
        "lag_1_date_pub": pd.to_datetime(["2022-10-25"]),
        "lag_2_date_pub": pd.to_datetime(["2022-07-26"]),
        "lag_3_date_pub": pd.to_datetime(["2022-04-26"]),
        "lag_4_date_pub": pd.to_datetime(["2022-01-25"]),
        "target_beat": [1],
    })

    return samples, events


def test_build_lagged_features():
    samples, events = make_feature_test_data()

    original_samples = samples.copy(deep=True)
    original_events = events.copy(deep=True)

    features = build_lagged_features(samples, events)

    assert features.shape == (1, 16)

    assert features.loc[0, "actual_eps_lag1"] == 0.90
    assert features.loc[0, "estimate_eps_lag1"] == 0.70
    assert features.loc[0, "surprise_eps_lag1"] == pytest.approx(0.20)
    assert features.loc[0, "beat_lag1"] == 1

    assert features.loc[0, "actual_eps_lag2"] == 0.60
    assert features.loc[0, "beat_lag2"] == 0

    assert features.loc[0, "actual_eps_lag3"] == -0.50
    assert features.loc[0, "surprise_eps_lag3"] == pytest.approx(-0.10)
    assert features.loc[0, "beat_lag3"] == 0

    assert features.loc[0, "actual_eps_lag4"] == 1.00
    assert features.loc[0, "beat_lag4"] == 1

    # targets outcomes and identifiers cannot enter the predictors
    assert "target_beat" not in features.columns
    assert "ticker" not in features.columns
    assert "target_date_pub" not in features.columns

    # target event has an artificial actual EPS of 99
    assert 99.0 not in features.to_numpy()

    pd.testing.assert_frame_equal(samples, original_samples)
    pd.testing.assert_frame_equal(events, original_events)


def test_lagged_features_preserve_missing_values():
    samples, events = make_feature_test_data()

    events.loc[1, "estimated_earning"] = None

    features = build_lagged_features(samples, events)

    assert features.loc[0, "actual_eps_lag3"] == -0.50
    assert pd.isna(features.loc[0, "estimate_eps_lag3"])
    assert pd.isna(features.loc[0, "surprise_eps_lag3"])
    assert pd.isna(features.loc[0, "beat_lag3"])


def test_lagged_features_reject_unknown_events():
    samples, events = make_feature_test_data()

    samples.loc[0, "lag_2_date_pub"] = pd.Timestamp("2022-07-25")

    with pytest.raises(ValueError, match="unknown earnings event"):
        build_lagged_features(samples, events)


def test_lagged_features_reject_same_day_history():
    samples, events = make_feature_test_data()

    samples.loc[0, "lag_1_date_pub"] = pd.Timestamp("2023-01-24")

    with pytest.raises(ValueError, match="strictly earlier"):
        build_lagged_features(samples, events)


def test_lagged_features_reject_duplicate_event_keys():
    samples, events = make_feature_test_data()

    duplicated_events = pd.concat(
        [events, events.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Duplicate earnings-event keys"):
        build_lagged_features(samples, duplicated_events)
        

def test_build_historical_features():
    samples, events = make_feature_test_data()

    features = build_historical_features(samples, events)

    assert features.shape == (1, 25)
    assert features.columns.tolist() == FEATURE_COLUMNS

    # summary statistics
    assert features.loc[0, "beat_rate4"] == 0.5
    assert features.loc[0, "surprise_mean4"] == pytest.approx(0.075)

    assert features.loc[0, "surprise_std4"] == pytest.approx(
        0.1299038106
    )

    # changes + trends
    assert features.loc[0, "eps_change1_2"] == pytest.approx(0.30)
    assert features.loc[0, "eps_trend4"] == pytest.approx(0.08)
    assert features.loc[0, "surprise_trend4"] == pytest.approx(0.01)

    # beat streak + calendar features
    assert features.loc[0, "beat_streak4"] == 1
    assert features.loc[0, "previous_report_month"] == 10
    assert features.loc[0, "previous_report_quarter"] == 4

    assert str(features["previous_report_month"].dtype) == "category"
    assert str(features["previous_report_quarter"].dtype) == "category"

    # final feature matrix excludes target information
    assert "target_beat" not in features.columns
    assert "ticker" not in features.columns
    assert "target_date_pub" not in features.columns


def test_derived_features_preserve_missing_inputs():
    samples, events = make_feature_test_data()

    # removes the analyst estimate for the third historical event
    events.loc[1, "estimated_earning"] = None

    features = build_historical_features(samples, events)

    missing_features = [
        "estimate_eps_lag3",
        "surprise_eps_lag3",
        "beat_lag3",
        "beat_rate4",
        "surprise_mean4",
        "surprise_std4",
        "surprise_trend4",
        "beat_streak4",
    ]

    for column in missing_features:
        assert pd.isna(features.loc[0, column])

    # actual EPS values remain available so its trend is valid
    assert features.loc[0, "eps_trend4"] == pytest.approx(0.08)
    assert features.loc[0, "eps_change1_2"] == pytest.approx(0.30)


def test_derived_features_reject_invalid_beat_indicators():
    samples, events = make_feature_test_data()

    lagged = build_lagged_features(samples, events)
    lagged.loc[0, "beat_lag1"] = 2

    with pytest.raises(ValueError, match="Beat indicators"):
        build_derived_features(samples, lagged)


def test_derived_features_require_matching_indices():
    samples, events = make_feature_test_data()

    lagged = build_lagged_features(samples, events)

    samples.index = [10]

    with pytest.raises(ValueError, match="indices must match"):
        build_derived_features(samples, lagged)
        

def test_main_generates_feature_artifacts(tmp_path, monkeypatch):
    samples, events = make_feature_test_data()

    event_path = tmp_path / EVENT_LEDGER_PATH
    sample_path = tmp_path / PREDICTION_WINDOWS_PATH

    event_path.parent.mkdir(parents=True, exist_ok=True)

    events.to_parquet(event_path, index=False)
    samples.to_parquet(sample_path, index=False)

    # runs the complete pipeline in an isolated directory
    monkeypatch.chdir(tmp_path)

    main()

    feature_path = tmp_path / FEATURE_MATRIX_PATH
    summary_path = tmp_path / FEATURE_SUMMARY_PATH

    assert feature_path.is_file()
    assert summary_path.is_file()

    saved_features = pd.read_parquet(feature_path)

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    assert saved_features.shape == (1, 25)
    assert saved_features.columns.tolist() == FEATURE_COLUMNS

    assert summary["prediction_samples"] == 1
    assert summary["feature_count"] == 25
    assert summary["numerical_features"] == 23
    assert summary["categorical_features"] == 2
    assert summary["rows_with_missing_features"] == 0

    # confirms that target info is excluded
    assert "target_beat" not in saved_features.columns
    assert "target_date_pub" not in saved_features.columns
    assert "ticker" not in saved_features.columns