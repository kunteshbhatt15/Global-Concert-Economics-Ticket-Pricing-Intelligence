from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from model_pipeline import (
    BINARY_MAP_COLUMNS,
    FREQUENCY_COLUMNS,
    ONE_HOT_COLUMNS,
    TARGET,
    load_artifacts,
    load_dataset,
    modeling_frame,
    predict_price,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "concert_dataset_cleaned.csv"
MODEL_DIR = BASE_DIR / "models"


st.set_page_config(
    page_title="Concert Ticket Pricing Intelligence",
    page_icon="Ã°Å¸Å½Å¸Ã¯Â¸Â",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --accent: #0f8b8d;
        --ink: #17202a;
        --muted: #5f6b7a;
        --surface: #f7f9fb;
        --line: #d8e0e8;
    }
    .stApp {
        background: linear-gradient(180deg, #fbfcfd 0%, #eef4f6 100%);
        color: var(--ink);
    }
    [data-testid="stSidebar"] {
        background: #102a43;
    }
    [data-testid="stSidebar"] * {
        color: #f6fbff;
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"],
    .stSlider label,
    .stSelectbox label,
    .stNumberInput label,
    .stCheckbox label {
        color: #17202a !important;
        font-weight: 750 !important;
        font-size: 0.94rem !important;
        line-height: 1.25 !important;
        opacity: 1 !important;
    }
    div[data-testid="stWidgetLabel"] {
        margin-bottom: 0.35rem;
    }
    [data-testid="stSidebar"] div[data-testid="stWidgetLabel"] label,
    [data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] label[data-testid="stWidgetLabel"] {
        color: #f6fbff !important;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(16, 42, 67, 0.05);
    }
    div[data-testid="stMetric"] label {
        color: var(--muted) !important;
        opacity: 1 !important;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] > div,
    div[data-testid="stMetricValue"] [data-testid="stMarkdownContainer"],
    div[data-testid="stMetricValue"] p {
        color: #17202a !important;
        opacity: 1 !important;
    }
    .sidebar-kpi {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 8px;
        padding: 10px 12px;
        margin: 8px 0;
    }
    .sidebar-kpi-label {
        color: rgba(246, 251, 255, 0.78);
        font-size: 0.78rem;
        line-height: 1.2;
    }
    .sidebar-kpi-value {
        color: #ffffff;
        font-size: 1.2rem;
        font-weight: 800;
        line-height: 1.25;
        margin-top: 3px;
    }
    .creator-box {
        background: rgba(15, 139, 141, 0.24);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        padding: 11px 12px;
        margin-top: 10px;
    }
    .creator-box strong {
        color: #ffffff;
    }
    .creator-box span {
        color: rgba(246, 251, 255, 0.82);
        font-size: 0.8rem;
    }
    .section-band {
        background: rgba(255, 255, 255, 0.78);
        border-top: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        padding: 18px 0 8px 0;
        margin: 12px 0 18px 0;
    }
    .price-callout {
        background: #ffffff;
        border-left: 5px solid var(--accent);
        border-radius: 8px;
        padding: 18px;
        box-shadow: 0 2px 8px rgba(16, 42, 67, 0.08);
    }
    .confidence-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border-radius: 999px;
        padding: 6px 11px;
        margin-top: 8px;
        font-weight: 800;
        font-size: 0.92rem;
    }
    .confidence-dot {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        display: inline-block;
    }
    .confidence-high {
        background: #e7f7ee;
        color: #126b37;
    }
    .confidence-high .confidence-dot {
        background: #1f9d55;
    }
    .confidence-medium {
        background: #fff7df;
        color: #8a5a00;
    }
    .confidence-medium .confidence-dot {
        background: #d99a00;
    }
    .confidence-low {
        background: #ffeceb;
        color: #9b1c1c;
    }
    .confidence-low .confidence-dot {
        background: #d64545;
    }
    .insight-box {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 16px 18px;
        margin-top: 14px;
        box-shadow: 0 1px 4px rgba(16, 42, 67, 0.05);
    }
    .insight-box h4 {
        margin: 0 0 10px 0;
        color: #17202a;
    }
    .insight-box li {
        margin-bottom: 6px;
        color: #25313f;
    }
    div[data-testid="stDownloadButton"] button,
    div[data-testid="stDownloadButton"] button *,
    div[data-testid="stDownloadButton"] button:hover,
    div[data-testid="stDownloadButton"] button:hover *,
    div[data-testid="stDownloadButton"] button:active,
    div[data-testid="stDownloadButton"] button:active * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    .small-muted {
        color: var(--muted);
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_dataset(DATA_PATH)


@st.cache_resource(show_spinner=False)
def get_artifacts() -> dict:
    return load_artifacts(MODEL_DIR)


def format_money(value: float) -> str:
    return f"${value:,.2f}"


def sidebar_kpi(label: str, value: str) -> None:
    st.sidebar.markdown(
        f"""
        <div class="sidebar-kpi">
            <div class="sidebar-kpi-label">{label}</div>
            <div class="sidebar-kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def option_list(df: pd.DataFrame, column: str) -> list:
    return sorted(df[column].dropna().astype(str).unique().tolist())


def default_index(options: list, preferred: str | None = None) -> int:
    if preferred and preferred in options:
        return options.index(preferred)
    return 0


def numeric_control(df: pd.DataFrame, column: str, label: str, step: float = 1.0) -> float:
    series = df[column]
    min_value = float(series.min())
    max_value = float(series.max())
    median = float(series.median())
    return st.number_input(label, min_value=min_value, max_value=max_value, value=median, step=step)


def estimate_uncertainty(input_df: pd.DataFrame, reference_df: pd.DataFrame) -> tuple[float, str, str]:
    row = input_df.iloc[0]
    checks = [
        ("spotify_popularity", 0.015),
        ("followers_millions", 0.012),
        ("monthly_listeners_millions", 0.012),
        ("total_streams_billions", 0.01),
        ("capacity", 0.012),
        ("num_artists_on_lineup", 0.008),
        ("competing_concerts_same_city_week", 0.012),
    ]
    interval_pct = 0.08
    for column, penalty in checks:
        q05 = reference_df[column].quantile(0.05)
        q95 = reference_df[column].quantile(0.95)
        if row[column] <= q05 or row[column] >= q95:
            interval_pct += penalty

    rare_columns = ["artist", "venue_name", "festival_name", "city", "country"]
    for column in rare_columns:
        share = (reference_df[column] == row[column]).mean()
        if share < 0.01:
            interval_pct += 0.008

    interval_pct = min(interval_pct, 0.22)
    if interval_pct <= 0.10:
        return interval_pct, "High confidence", "high"
    if interval_pct <= 0.15:
        return interval_pct, "Medium confidence", "medium"
    return interval_pct, "Low confidence", "low"


def explain_prediction(input_df: pd.DataFrame, reference_df: pd.DataFrame) -> list[str]:
    row = input_df.iloc[0]
    reasons = []
    if row["spotify_popularity"] >= reference_df["spotify_popularity"].quantile(0.75):
        reasons.append("High Spotify popularity is pushing the predicted ticket price upward.")
    if row["followers_millions"] >= reference_df["followers_millions"].quantile(0.75):
        reasons.append("Large follower count indicates stronger demand potential.")
    if row["monthly_listeners_millions"] >= reference_df["monthly_listeners_millions"].quantile(0.75):
        reasons.append("High monthly listeners signal a broad active audience.")
    if row["capacity"] >= reference_df["capacity"].quantile(0.75):
        reasons.append("Large venue capacity supports premium event economics.")
    if row["is_headliner"] == "Yes" or row["billing_status"] == "Headliner":
        reasons.append("Headliner billing usually commands a higher average ticket price.")
    if row["festival_tier"] >= 4:
        reasons.append("Higher festival tier is associated with stronger pricing power.")
    if row["peak_season"] == 1 or row["is_weekend"] == 1:
        reasons.append("Weekend or peak-season timing can improve expected demand.")
    if row["competing_concerts_same_city_week"] >= reference_df["competing_concerts_same_city_week"].quantile(0.75):
        reasons.append("Competing concerts in the same city/week may add pricing uncertainty.")

    if not reasons:
        reasons.append("The prediction is mainly driven by the combined artist, venue, genre, and calendar profile.")
    return reasons[:5]


def build_prediction_report(
    input_df: pd.DataFrame,
    prediction: float,
    low: float,
    high: float,
    confidence_label: str,
    interval_pct: float,
    reasons: list[str],
) -> bytes:
    report = input_df.copy()
    report.insert(0, "predicted_average_ticket_price_usd", round(prediction, 2))
    report.insert(1, "prediction_lower_bound_usd", round(low, 2))
    report.insert(2, "prediction_upper_bound_usd", round(high, 2))
    report.insert(3, "confidence", confidence_label)
    report.insert(4, "interval_width_pct", round(interval_pct * 100, 2))
    report.insert(5, "prediction_explanation", " | ".join(reasons))
    return report.to_csv(index=False).encode("utf-8")


def feature_label(name: str, column: str) -> str:
    return f"{name} ({column})"


def build_prediction_input(df: pd.DataFrame) -> pd.DataFrame:
    model_df = modeling_frame(df)
    feature_df = model_df.drop(columns=[TARGET])
    sample = feature_df.sample(1, random_state=7).iloc[0].to_dict()

    st.subheader("Event Profile")
    left, middle, right = st.columns(3)

    with left:
        sample["artist"] = st.selectbox(feature_label("Artist", "artist"), option_list(feature_df, "artist"))
        sample["artist_country"] = st.selectbox(feature_label("Artist country", "artist_country"), option_list(feature_df, "artist_country"))
        sample["primary_genre"] = st.selectbox(feature_label("Primary genre", "primary_genre"), option_list(feature_df, "primary_genre"))
        sample["secondary_genre"] = st.selectbox(feature_label("Secondary genre", "secondary_genre"), option_list(feature_df, "secondary_genre"))
        sample["record_label"] = st.selectbox(feature_label("Record label", "record_label"), option_list(feature_df, "record_label"))
        sample["multi_genre_artist"] = st.segmented_control(
            feature_label("Multi-genre artist", "multi_genre_artist"), ["Yes", "No"], default=str(sample["multi_genre_artist"])
        )
        sample["grammy_winner"] = int(st.toggle(feature_label("Grammy winner", "grammy_winner"), value=bool(sample["grammy_winner"])))

    with middle:
        sample["event_type"] = st.selectbox(feature_label("Event type", "event_type"), option_list(feature_df, "event_type"))
        sample["event_status"] = st.selectbox(feature_label("Event status", "event_status"), option_list(feature_df, "event_status"))
        sample["festival_name"] = st.selectbox(feature_label("Festival", "festival_name"), option_list(feature_df, "festival_name"))
        sample["festival_tier"] = st.slider(feature_label("Festival tier", "festival_tier"), 1, 5, int(sample["festival_tier"]))
        sample["num_artists_on_lineup"] = st.slider(
            feature_label("Artists on lineup", "num_artists_on_lineup"),
            int(feature_df["num_artists_on_lineup"].min()),
            int(feature_df["num_artists_on_lineup"].max()),
            int(sample["num_artists_on_lineup"]),
        )
        sample["billing_status"] = st.selectbox(feature_label("Billing status", "billing_status"), option_list(feature_df, "billing_status"))
        sample["is_headliner"] = st.segmented_control(feature_label("Headliner", "is_headliner"), ["Yes", "No"], default=str(sample["is_headliner"]))
        sample["is_sub_headliner"] = st.segmented_control(
            feature_label("Sub-headliner", "is_sub_headliner"), ["Yes", "No"], default=str(sample["is_sub_headliner"])
        )

    with right:
        sample["venue_name"] = st.selectbox(feature_label("Venue", "venue_name"), option_list(feature_df, "venue_name"))
        sample["venue_type"] = st.selectbox(feature_label("Venue type", "venue_type"), option_list(feature_df, "venue_type"))
        sample["city"] = st.selectbox(feature_label("City", "city"), option_list(feature_df, "city"))
        sample["country"] = st.selectbox(feature_label("Country", "country"), option_list(feature_df, "country"))
        sample["capacity"] = st.number_input(
            feature_label("Venue capacity", "capacity"),
            min_value=int(feature_df["capacity"].min()),
            max_value=int(feature_df["capacity"].max()),
            value=int(sample["capacity"]),
            step=500,
        )
        sample["distance_from_airport_km"] = st.slider(
            feature_label("Distance from airport in km", "distance_from_airport_km"),
            float(feature_df["distance_from_airport_km"].min()),
            float(feature_df["distance_from_airport_km"].max()),
            float(sample["distance_from_airport_km"]),
            step=0.1,
        )

    st.subheader("Market Signals")
    a, b, c, d = st.columns(4)
    with a:
        sample["spotify_popularity"] = st.slider(feature_label("Spotify popularity", "spotify_popularity"), 15, 100, int(sample["spotify_popularity"]))
        sample["followers_millions"] = numeric_control(feature_df, "followers_millions", feature_label("Followers in millions", "followers_millions"), 0.1)
        sample["monthly_listeners_millions"] = numeric_control(
            feature_df, "monthly_listeners_millions", feature_label("Monthly listeners in millions", "monthly_listeners_millions"), 0.1
        )
    with b:
        sample["total_streams_billions"] = numeric_control(
            feature_df, "total_streams_billions", feature_label("Total streams in billions", "total_streams_billions"), 0.1
        )
        sample["years_active"] = st.slider(
            feature_label("Years active", "years_active"),
            int(feature_df["years_active"].min()),
            int(feature_df["years_active"].max()),
            int(sample["years_active"]),
        )
        sample["google_trend_index"] = st.slider(feature_label("Google trend index", "google_trend_index"), 0, 100, int(sample["google_trend_index"]))
    with c:
        sample["instagram_followers_millions"] = numeric_control(
            feature_df, "instagram_followers_millions", feature_label("Instagram followers in millions", "instagram_followers_millions"), 0.1
        )
        sample["tiktok_followers_millions"] = numeric_control(
            feature_df, "tiktok_followers_millions", feature_label("TikTok followers in millions", "tiktok_followers_millions"), 0.1
        )
        sample["youtube_subscribers_millions"] = numeric_control(
            feature_df, "youtube_subscribers_millions", feature_label("YouTube subscribers in millions", "youtube_subscribers_millions"), 0.1
        )
    with d:
        sample["twitter_followers_millions"] = numeric_control(
            feature_df, "twitter_followers_millions", feature_label("Twitter/X followers in millions", "twitter_followers_millions"), 0.1
        )
        sample["facebook_followers_millions"] = numeric_control(
            feature_df, "facebook_followers_millions", feature_label("Facebook followers in millions", "facebook_followers_millions"), 0.1
        )
        sample["competing_concerts_same_city_week"] = st.slider(
            feature_label("Competing concerts same city/week", "competing_concerts_same_city_week"),
            int(feature_df["competing_concerts_same_city_week"].min()),
            int(feature_df["competing_concerts_same_city_week"].max()),
            int(sample["competing_concerts_same_city_week"]),
        )

    st.subheader("Calendar And Conditions")
    cal1, cal2, cal3, cal4 = st.columns(4)
    with cal1:
        sample["day_of_week"] = st.selectbox(feature_label("Day of week", "day_of_week"), option_list(feature_df, "day_of_week"))
        sample["month"] = st.slider(feature_label("Month", "month"), 1, 12, int(sample["month"]))
    with cal2:
        sample["quarter"] = st.slider(feature_label("Quarter", "quarter"), 1, 4, int(sample["quarter"]))
        sample["year"] = st.slider(feature_label("Year", "year"), int(feature_df["year"].min()), int(feature_df["year"].max()), int(sample["year"]))
    with cal3:
        sample["is_weekend"] = int(st.toggle(feature_label("Weekend", "is_weekend"), value=bool(sample["is_weekend"])))
        sample["is_holiday"] = int(st.toggle(feature_label("Holiday", "is_holiday"), value=bool(sample["is_holiday"])))
    with cal4:
        sample["peak_season"] = int(st.toggle(feature_label("Peak season", "peak_season"), value=bool(sample["peak_season"])))
        sample["weather_type"] = st.selectbox(feature_label("Weather type", "weather_type"), option_list(feature_df, "weather_type"))
        sample["Covid"] = st.selectbox(feature_label("Covid period", "Covid"), option_list(feature_df, "Covid"))

    return pd.DataFrame([sample], columns=feature_df.columns)


def overview(df: pd.DataFrame, artifacts: dict) -> None:
    st.title("Concert Ticket Pricing Intelligence")
    st.caption("Tuned XGBoost regression model for global concert average ticket price prediction.")

    metric_cols = st.columns(5)
    metrics = artifacts["metrics"]
    metric_cols[0].metric("Events", f"{len(df):,}")
    metric_cols[1].metric("Avg ticket price", format_money(df[TARGET].mean()))
    metric_cols[2].metric("Tuned XGB R2", f"{metrics['Test R2']:.3f}")
    metric_cols[3].metric("MAE", format_money(metrics["MAE"]))
    metric_cols[4].metric("RMSE", format_money(metrics["RMSE"]))

    left, right = st.columns((1.25, 1))
    with left:
        genre = df.groupby("primary_genre", as_index=False)[TARGET].mean().sort_values(TARGET, ascending=False)
        st.plotly_chart(
            px.bar(
                genre,
                x="primary_genre",
                y=TARGET,
                color=TARGET,
                color_continuous_scale="Teal",
                labels={TARGET: "Average ticket price", "primary_genre": "Genre"},
            ),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            px.histogram(
                df,
                x=TARGET,
                nbins=40,
                color_discrete_sequence=["#0f8b8d"],
                labels={TARGET: "Average ticket price"},
            ),
            width="stretch",
        )

    st.plotly_chart(
        px.scatter(
            df.sample(min(3000, len(df)), random_state=12),
            x="monthly_listeners_millions",
            y=TARGET,
            color="billing_status",
            size="capacity",
            hover_data=["artist", "city", "country", "primary_genre"],
            labels={"monthly_listeners_millions": "Monthly listeners (millions)", TARGET: "Average ticket price"},
        ),
        width="stretch",
    )


def predictor(df: pd.DataFrame, artifacts: dict) -> None:
    st.title("Price Predictor")
    st.caption("Estimate an average ticket price using the trained preprocessing pipeline and tuned XGBoost model.")
    input_df = build_prediction_input(df)
    prediction = float(predict_price(input_df, artifacts)[0])

    interval_pct, confidence_label, confidence_level = estimate_uncertainty(input_df, modeling_frame(df))
    low = prediction * (1 - interval_pct)
    high = prediction * (1 + interval_pct)
    reasons = explain_prediction(input_df, modeling_frame(df))
    report_bytes = build_prediction_report(input_df, prediction, low, high, confidence_label, interval_pct, reasons)
    reason_items = "".join(f"<li>{reason}</li>" for reason in reasons)
    st.markdown(
        f"""
        <div class="price-callout">
            <div class="small-muted">Predicted average ticket price</div>
            <h2>{format_money(prediction)}</h2>
            <div class="small-muted">Planning band: {format_money(low)} to {format_money(high)} ({interval_pct * 100:.1f}% interval)</div>
            <div class="confidence-badge confidence-{confidence_level}">
                <span class="confidence-dot"></span>
                <span>{confidence_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="insight-box">
            <h4>Prediction explanation</h4>
            <ul>{reason_items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.download_button(
        "Download prediction report",
        data=report_bytes,
        file_name="concert_ticket_prediction_report.csv",
        mime="text/csv",
        width="stretch",
    )

    with st.expander("Encoded model input"):
        st.caption("SHAP single-prediction contribution can be added here later for a true waterfall plot.")
        st.dataframe(input_df, width="stretch", hide_index=True)


def what_if(df: pd.DataFrame, artifacts: dict) -> None:
    st.title("What-If Simulator")
    model_df = modeling_frame(df)
    feature_df = model_df.drop(columns=[TARGET])
    base_row = feature_df.sample(1, random_state=22).iloc[0].to_dict()

    col1, col2, col3 = st.columns(3)
    with col1:
        base_row["artist"] = st.selectbox("Artist", option_list(feature_df, "artist"))
        base_row["primary_genre"] = st.selectbox("Genre", option_list(feature_df, "primary_genre"))
        base_row["billing_status"] = st.selectbox("Billing", option_list(feature_df, "billing_status"))
    with col2:
        capacity = st.slider(
            "Venue capacity",
            int(feature_df["capacity"].min()),
            int(feature_df["capacity"].max()),
            int(base_row["capacity"]),
            step=1000,
        )
        popularity = st.slider("Spotify popularity", 15, 100, int(base_row["spotify_popularity"]))
    with col3:
        listeners = st.slider(
            "Monthly listeners (millions)",
            float(feature_df["monthly_listeners_millions"].min()),
            float(feature_df["monthly_listeners_millions"].max()),
            float(base_row["monthly_listeners_millions"]),
            step=0.5,
        )
        competitors = st.slider(
            "Competing concerts",
            int(feature_df["competing_concerts_same_city_week"].min()),
            int(feature_df["competing_concerts_same_city_week"].max()),
            int(base_row["competing_concerts_same_city_week"]),
        )

    scenarios = []
    for label, cap_mult, pop_delta, listener_mult in [
        ("Conservative", 0.75, -8, 0.75),
        ("Base", 1.0, 0, 1.0),
        ("High demand", 1.25, 8, 1.35),
    ]:
        row = base_row.copy()
        row["capacity"] = int(np.clip(capacity * cap_mult, feature_df["capacity"].min(), feature_df["capacity"].max()))
        row["spotify_popularity"] = int(np.clip(popularity + pop_delta, 15, 100))
        row["monthly_listeners_millions"] = float(
            np.clip(
                listeners * listener_mult,
                feature_df["monthly_listeners_millions"].min(),
                feature_df["monthly_listeners_millions"].max(),
            )
        )
        row["competing_concerts_same_city_week"] = competitors
        price = float(predict_price(pd.DataFrame([row], columns=feature_df.columns), artifacts)[0])
        scenarios.append({"Scenario": label, "Predicted price": price})

    scenario_df = pd.DataFrame(scenarios)
    st.plotly_chart(
        px.bar(
            scenario_df,
            x="Scenario",
            y="Predicted price",
            color="Scenario",
            color_discrete_sequence=["#5b8e7d", "#0f8b8d", "#d96c06"],
            text=scenario_df["Predicted price"].map(format_money),
        ),
        width="stretch",
    )
    st.dataframe(scenario_df.assign(**{"Predicted price": scenario_df["Predicted price"].map(format_money)}), hide_index=True)


def model_intelligence(artifacts: dict) -> None:
    st.title("Model Intelligence")
    metrics = pd.DataFrame(
        [{"Metric": key, "Value": value} for key, value in artifacts["metrics"].items()]
    )
    st.dataframe(metrics, hide_index=True, width="stretch")

    importance = artifacts["feature_importance"].head(20)
    st.plotly_chart(
        px.bar(
            importance.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Tealgrn",
        ),
        width="stretch",
    )


def data_explorer(df: pd.DataFrame) -> None:
    st.title("Data Explorer")
    c1, c2, c3 = st.columns(3)
    genre = c1.multiselect("Genre", option_list(df, "primary_genre"), default=[])
    country = c2.multiselect("Country", option_list(df, "country"), default=[])
    billing = c3.multiselect("Billing", option_list(df, "billing_status"), default=[])

    filtered = df.copy()
    if genre:
        filtered = filtered[filtered["primary_genre"].isin(genre)]
    if country:
        filtered = filtered[filtered["country"].isin(country)]
    if billing:
        filtered = filtered[filtered["billing_status"].isin(billing)]

    st.metric("Filtered events", f"{len(filtered):,}")
    st.dataframe(filtered, width="stretch", height=520)


def main() -> None:
    if not DATA_PATH.exists():
        st.error(f"Dataset not found at {DATA_PATH}")
        st.stop()
    if not (MODEL_DIR / "model.pkl").exists():
        st.error("Model artifacts are missing. Run `python train_model.py` first.")
        st.stop()

    df = get_data()
    artifacts = get_artifacts()
    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Price Predictor", "What-If Simulator", "Model Intelligence", "Data Explorer"],
    )
    st.sidebar.markdown("---")
    metrics = artifacts["metrics"]
    sidebar_kpi("Events", f"{len(df):,}")
    sidebar_kpi("Avg ticket price", format_money(df[TARGET].mean()))
    sidebar_kpi("Tuned XGB R2", f"{metrics['Test R2']:.3f}")
    sidebar_kpi("MAE", format_money(metrics["MAE"]))
    sidebar_kpi("RMSE", format_money(metrics["RMSE"]))
    st.sidebar.markdown("---")
    st.sidebar.caption("Final model: tuned XGBoost regression")
    st.sidebar.caption("Target: average_ticket_price_usd")
    st.sidebar.markdown(
        """
        <div class="creator-box">
            <strong>Created by Kuntesh Bhatt</strong><br>
            <span>MTech Data Science and AI</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if page == "Overview":
        overview(df, artifacts)
    elif page == "Price Predictor":
        predictor(df, artifacts)
    elif page == "What-If Simulator":
        what_if(df, artifacts)
    elif page == "Model Intelligence":
        model_intelligence(artifacts)
    else:
        data_explorer(df)


if __name__ == "__main__":
    main()

