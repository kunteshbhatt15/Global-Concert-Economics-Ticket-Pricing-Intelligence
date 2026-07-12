from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor


TARGET = "average_ticket_price_usd"

LEAKAGE_AND_UNUSED_COLUMNS = [
    "early_bird_price_usd",
    "ga_price_usd",
    "vip_price_usd",
    "vvip_price_usd",
    "dynamic_price_factor",
    "merchandise_revenue_usd",
    "parking_revenue_usd",
    "food_revenue_usd",
    "bar_revenue_usd",
    "sponsor_revenue_usd",
    "advertising_revenue_usd",
    "avg_spend_per_customer_usd",
    "total_revenue_usd",
    "profit_usd",
    "date",
    "attendance",
    "sellout_pct",
    "occupancy_pct",
    "artist_age",
    "temperature_c",
    "humidity_pct",
    "rainfall_mm",
    "wind_kmh",
    "cloud_cover_pct",
    "heat_index",
    "feels_like_c",
    "uv_index",
]

# Columns that are perfectly (or near-perfectly) redundant with other features
# already in the model -- confirmed via VIF analysis in the training notebook:
#   - is_headliner / is_sub_headliner are exact duplicates of the one-hot
#     billing_status dummies (billing_status == 'Headliner' / 'Sub-Headliner')
#   - quarter is a deterministic function of month
#   - is_weekend is a deterministic function of day_of_week
# These are dropped BEFORE encoding/scaling rather than after, which keeps
# this pipeline single-pass (fit the scaler once, on the final feature set --
# no need to track two different column orders like the notebook's
# scale-then-drop sequence required).
REDUNDANT_DERIVED_COLUMNS = [
    "is_headliner",
    "is_sub_headliner",
    "quarter",
    "is_weekend",
]

# event_type_Festival (produced by one-hot encoding event_type) is ~0.9999
# correlated with festival_name's frequency encoding -- almost every row with
# a real festival_name also has event_type == 'Festival'. Dropped post-encoding
# since it only exists once the OneHotEncoder creates it.
OHE_COLUMNS_TO_DROP_POST_ENCODING = ["event_type_Festival"]

FREQUENCY_COLUMNS = [
    "artist",
    "artist_country",
    "venue_name",
    "festival_name",
    "city",
    "secondary_genre",
    "country",
]

ONE_HOT_COLUMNS = [
    "primary_genre",
    "day_of_week",
    "weather_type",
    "record_label",
    "event_type",
    "billing_status",
    "Covid",
]

BINARY_MAP_COLUMNS = {
    "multi_genre_artist": {"Yes": 1, "No": 0},
    "event_status": {"Held": 1, "Cancelled": 0},
}

ORDINAL_COLUMNS = ["venue_type"]
CLIPPED_COLUMN = "competing_concerts_same_city_week"


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = LEAKAGE_AND_UNUSED_COLUMNS + REDUNDANT_DERIVED_COLUMNS
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


def _make_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)


def fit_artifacts(df: pd.DataFrame) -> dict[str, Any]:
    model_df = modeling_frame(df)
    x = model_df.drop(columns=[TARGET])
    y = model_df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

    x_train = x_train.copy()
    x_test = x_test.copy()

    frequency_maps = {}
    for column in FREQUENCY_COLUMNS:
        mapping = x_train[column].value_counts()
        frequency_maps[column] = mapping
        x_train[column] = x_train[column].map(mapping)
        x_test[column] = x_test[column].map(mapping).fillna(0)

    onehot_encoder = _make_encoder()
    onehot_encoder.fit(x_train[ONE_HOT_COLUMNS])
    x_train_ohe = pd.DataFrame(
        onehot_encoder.transform(x_train[ONE_HOT_COLUMNS]),
        columns=onehot_encoder.get_feature_names_out(ONE_HOT_COLUMNS),
        index=x_train.index,
    )
    x_test_ohe = pd.DataFrame(
        onehot_encoder.transform(x_test[ONE_HOT_COLUMNS]),
        columns=onehot_encoder.get_feature_names_out(ONE_HOT_COLUMNS),
        index=x_test.index,
    )
    x_train = pd.concat([x_train.drop(columns=ONE_HOT_COLUMNS), x_train_ohe], axis=1)
    x_test = pd.concat([x_test.drop(columns=ONE_HOT_COLUMNS), x_test_ohe], axis=1)
    x_train = x_train.drop(columns=[c for c in OHE_COLUMNS_TO_DROP_POST_ENCODING if c in x_train.columns])
    x_test = x_test.drop(columns=[c for c in OHE_COLUMNS_TO_DROP_POST_ENCODING if c in x_test.columns])

    ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    x_train[ORDINAL_COLUMNS] = ordinal_encoder.fit_transform(x_train[ORDINAL_COLUMNS])
    x_test[ORDINAL_COLUMNS] = ordinal_encoder.transform(x_test[ORDINAL_COLUMNS])

    for column, mapping in BINARY_MAP_COLUMNS.items():
        x_train[column] = x_train[column].map(mapping)
        x_test[column] = x_test[column].map(mapping)

    q1 = x_train[CLIPPED_COLUMN].quantile(0.25)
    q3 = x_train[CLIPPED_COLUMN].quantile(0.75)
    iqr = q3 - q1
    clipping_bounds = {
        CLIPPED_COLUMN: {
            "lower_fence": q1 - 1.5 * iqr,
            "upper_fence": q3 + 1.5 * iqr,
        }
    }
    bounds = clipping_bounds[CLIPPED_COLUMN]
    x_train[CLIPPED_COLUMN] = x_train[CLIPPED_COLUMN].clip(bounds["lower_fence"], bounds["upper_fence"])
    x_test[CLIPPED_COLUMN] = x_test[CLIPPED_COLUMN].clip(bounds["lower_fence"], bounds["upper_fence"])

    scaler = StandardScaler()
    feature_columns = x_train.columns.tolist()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=feature_columns, index=x_train.index)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=feature_columns, index=x_test.index)

    model = XGBRegressor(
        subsample=0.9,
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train_scaled, y_train)
    predictions = model.predict(x_test_scaled)

    r2 = r2_score(y_test, predictions)
    n = len(y_test)
    p = x_test_scaled.shape[1]
    metrics = {
        "R2": r2,
        "Adjusted R2": 1 - (((1 - r2) * (n - 1)) / (n - p - 1)),
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "MAPE": mean_absolute_percentage_error(y_test, predictions),
        "Train R2": model.score(x_train_scaled, y_train),
        "Test R2": r2,
    }

    feature_importance = (
        pd.DataFrame({"Feature": feature_columns, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "model": model,
        "frequency_maps": frequency_maps,
        "onehot_encoder": onehot_encoder,
        "ordinal_encoder": ordinal_encoder,
        "scaler": scaler,
        "clipping_bounds": clipping_bounds,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "feature_importance": feature_importance,
    }


def transform_input(input_df: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    x = input_df.copy()
    for column in FREQUENCY_COLUMNS:
        mapping = artifacts["frequency_maps"][column]
        x[column] = x[column].map(mapping).fillna(0)

    ohe = artifacts["onehot_encoder"]
    ohe_df = pd.DataFrame(
        ohe.transform(x[ONE_HOT_COLUMNS]),
        columns=ohe.get_feature_names_out(ONE_HOT_COLUMNS),
        index=x.index,
    )
    x = pd.concat([x.drop(columns=ONE_HOT_COLUMNS), ohe_df], axis=1)
    x = x.drop(columns=[c for c in OHE_COLUMNS_TO_DROP_POST_ENCODING if c in x.columns])
    x[ORDINAL_COLUMNS] = artifacts["ordinal_encoder"].transform(x[ORDINAL_COLUMNS])

    for column, mapping in BINARY_MAP_COLUMNS.items():
        x[column] = x[column].map(mapping)

    bounds = artifacts["clipping_bounds"][CLIPPED_COLUMN]
    x[CLIPPED_COLUMN] = x[CLIPPED_COLUMN].clip(bounds["lower_fence"], bounds["upper_fence"])
    x = x.reindex(columns=artifacts["feature_columns"], fill_value=0)
    return pd.DataFrame(artifacts["scaler"].transform(x), columns=artifacts["feature_columns"], index=x.index)


def predict_price(input_df: pd.DataFrame, artifacts: dict[str, Any]) -> np.ndarray:
    transformed = transform_input(input_df, artifacts)
    return artifacts["model"].predict(transformed)


def save_artifacts(artifacts: dict[str, Any], model_dir: str | Path) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for key, value in artifacts.items():
        joblib.dump(value, model_dir / f"{key}.pkl")


def load_artifacts(model_dir: str | Path) -> dict[str, Any]:
    model_dir = Path(model_dir)
    return {
        "model": joblib.load(model_dir / "xgboost_model.pkl"),
        "frequency_maps": joblib.load(model_dir / "frequency_maps.pkl"),
        "onehot_encoder": joblib.load(model_dir / "onehot_encoder.pkl"),
        "ordinal_encoder": joblib.load(model_dir / "venue_ordinal_encoder.pkl"),
        "scaler": joblib.load(model_dir / "standard_scaler.pkl"),
        "clipping_bounds": joblib.load(model_dir / "clipping_bounds.pkl"),
        "feature_columns": joblib.load(model_dir / "feature_columns.pkl"),
        "metrics": joblib.load(model_dir / "metrics.pkl"),
        "feature_importance": joblib.load(model_dir / "feature_importance.pkl"),
    }
