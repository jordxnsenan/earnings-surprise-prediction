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
    column
    for column in FEATURE_COLUMNS
    if column not in CATEGORICAL_COLUMNS + DISCRETE_COLUMNS
]

MONTH_CATEGORIES = list(range(1, 13))
QUARTER_CATEGORIES = list(range(1, 5))


def build_preprocessor(
    feature_columns: list[str] | None = None,
) -> ColumnTransformer:
    """creates a preprocessing pipeline for the selected features"""
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    if not feature_columns:
        raise ValueError("At least one feature is required")

    unknown_columns = [
        column for column in feature_columns if column not in FEATURE_COLUMNS
    ]

    if unknown_columns:
        raise ValueError(f"Unknown feature columns: {unknown_columns}")

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Duplicate feature columns")

    continuous_columns = [
        column for column in feature_columns if column in CONTINUOUS_COLUMNS
    ]

    discrete_columns = [
        column for column in feature_columns if column in DISCRETE_COLUMNS
    ]

    categorical_columns = [
        column for column in feature_columns if column in CATEGORICAL_COLUMNS
    ]

    continuous_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    discrete_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    calendar_categories = {
        "previous_report_month": MONTH_CATEGORIES,
        "previous_report_quarter": QUARTER_CATEGORIES,
    }

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    categories=[
                        calendar_categories[column] for column in categorical_columns
                    ],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("continuous", continuous_pipeline, continuous_columns),
            ("discrete", discrete_pipeline, discrete_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_transformed_feature_names() -> list[str]:
    """returns expected names of the preprocessed feature columns"""
    feature_names = CONTINUOUS_COLUMNS + DISCRETE_COLUMNS

    calendar_categories = [
        MONTH_CATEGORIES,
        QUARTER_CATEGORIES,
    ]

    for column, categories in zip(
        CATEGORICAL_COLUMNS,
        calendar_categories,
        strict=True,
    ):
        for category in categories:
            feature_names.append(f"{column}_{category}")

    return feature_names
