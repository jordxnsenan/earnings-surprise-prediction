from earnings_surprise.features import FEATURE_COLUMNS

LATEST_EVENT_FEATURES = [
    "actual_eps_lag1",
    "estimate_eps_lag1",
    "surprise_eps_lag1",
    "beat_lag1",
    "previous_report_month",
    "previous_report_quarter",
]

TREND_FEATURES = [
    "eps_change1_2",
    "eps_trend4",
    "surprise_trend4",
]

FEATURE_SETS = {
    "full_history": FEATURE_COLUMNS,
    "latest_event_only": LATEST_EVENT_FEATURES,
    "without_trends": [
        column for column in FEATURE_COLUMNS if column not in TREND_FEATURES
    ],
}
