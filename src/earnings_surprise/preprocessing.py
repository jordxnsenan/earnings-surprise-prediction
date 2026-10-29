from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from earnings_surprise.features import FEATURE_COLUMNS


CATEGORICAL_COLUMNS = [
    "previous_report_month",
    "previous_report_quarter",
]

DISCRETE_COLUMNS = [
    "beat_lag1",
    "beat_lag2",
    "beat_lag3",
    "beat_lag4",
    "beat_rate4",
    "beat_streak4",
]

CONTINUOUS_COLUMNS = [
    column for column in FEATURE_COLUMNS
    if column not in CATEGORICAL_COLUMNS + DISCRETE_COLUMNS
]

MONTH_CATEGORIES = list(range(1, 13))
QUARTER_CATEGORIES = list(range(1, 5))


def build_preprocessor() -> ColumnTransformer:
    """Create a new preprocessing pipeline for historical earnings features."""
    continuous_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    discrete_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            categories=[
                MONTH_CATEGORIES,
                QUARTER_CATEGORIES,
            ],
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("continuous", continuous_pipeline, CONTINUOUS_COLUMNS),
            ("discrete", discrete_pipeline, DISCRETE_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )