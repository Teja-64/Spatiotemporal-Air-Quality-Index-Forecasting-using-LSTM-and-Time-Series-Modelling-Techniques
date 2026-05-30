import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from tensorflow.keras.models import load_model


DATA_PATH = "data/processed/processed_data.csv"
METRICS_PATH = "models/metrics.json"
SCALER_PATH = "artifacts/scaler.joblib"
WD_ENCODER_PATH = "artifacts/wd_encoder.joblib"
STATION_ENCODER_PATH = "artifacts/station_encoder.joblib"
FEATURE_CONFIG_PATH = "artifacts/feature_config.json"

MODEL_PATHS = {
    "LSTM": "models/best_lstm_model.keras",
    "BiLSTM": "models/best_bilstm_model.keras",
    "GRU": "models/best_gru_model.keras",
    "BiLSTM+Conv1D": "models/best_bilstm_conv1d_model.keras",
}

INDIAN_STATION_LABELS = [
    "Delhi NCR - Anand Vihar",
    "Delhi NCR - Ashok Vihar",
    "Delhi NCR - DTU",
    "Delhi NCR - Mandir Marg",
    "Delhi NCR - Rohini",
    "Delhi NCR - Vivek Vihar",
    "Hyderabad - Central University",
    "Hyderabad - ICRISAT Patancheru",
    "Hyderabad - Bollaram Industrial Area",
    "Hyderabad - IDA Pashamylaram",
    "Hyderabad - Zoo Park",
    "Hyderabad - Sanathnagar",
]

st.set_page_config(
    page_title="Spatiotemporal AQI Forecasting",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def local_css(file_name: str) -> None:
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as css_file:
            st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)


def encode_cyclical(value: float, period: int) -> tuple[float, float]:
    radians = 2 * np.pi * float(value) / period
    return float(np.sin(radians)), float(np.cos(radians))


def pm25_to_aqi(pm25_ugm3: float):
    breakpoints = [
        (0.0, 12.0, 0, 50, "Good", "#00e400"),
        (12.1, 35.4, 51, 100, "Moderate", "#ffff00"),
        (35.5, 55.4, 101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
        (55.5, 150.4, 151, 200, "Unhealthy", "#ff0000"),
        (150.5, 250.4, 201, 300, "Very Unhealthy", "#8f3f97"),
        (250.5, 500.4, 301, 500, "Hazardous", "#7e0023"),
    ]
    for c_lo, c_hi, i_lo, i_hi, label, color in breakpoints:
        if c_lo <= pm25_ugm3 <= c_hi:
            aqi = round(((i_hi - i_lo) / (c_hi - c_lo)) * (pm25_ugm3 - c_lo) + i_lo)
            return aqi, label, color
    if pm25_ugm3 < 0:
        return 0, "Good", "#00e400"
    return 500, "Hazardous", "#7e0023"


def parse_history_values(raw_text: str, seq_length: int):
    if not raw_text.strip():
        return None
    try:
        values = [float(item.strip()) for item in raw_text.split(",") if item.strip()]
    except ValueError:
        return "invalid"
    if len(values) != seq_length:
        return "invalid"
    return values


def build_station_display_map(station_names: list[str]) -> dict[str, str]:
    display_map = {}
    for index, station_name in enumerate(station_names):
        alias = INDIAN_STATION_LABELS[index] if index < len(INDIAN_STATION_LABELS) else f"India Demo Station {index + 1}"
        display_map[station_name] = f"{alias} [{station_name}]"
    return display_map


@st.cache_resource
def load_resources():
    try:
        scaler = joblib.load(SCALER_PATH)
        wd_encoder = joblib.load(WD_ENCODER_PATH)
        station_encoder = joblib.load(STATION_ENCODER_PATH)
        with open(FEATURE_CONFIG_PATH, "r", encoding="utf-8") as config_file:
            feature_config = json.load(config_file)
        return scaler, wd_encoder, station_encoder, feature_config
    except Exception:
        return None, None, None, None


@st.cache_resource
def load_prediction_model(model_path: str):
    if not os.path.exists(model_path):
        return None
    return load_model(model_path, compile=False)


@st.cache_data
def load_processed_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH, parse_dates=["datetime"])


@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as metrics_file:
            return json.load(metrics_file)
    return None


def inverse_pm25_from_scaled(pred_scaled: float, scaler, numeric_columns: list[str]) -> float:
    row = np.zeros((1, len(numeric_columns)), dtype=float)
    row[0, 0] = pred_scaled
    value = float(scaler.inverse_transform(row)[0, 0])
    return max(0.0, value)


def inverse_pm25_series(values: pd.Series, scaler, numeric_columns: list[str]) -> np.ndarray:
    rows = np.zeros((len(values), len(numeric_columns)), dtype=float)
    rows[:, 0] = values.to_numpy(dtype=float)
    return scaler.inverse_transform(rows)[:, 0]


def build_context_vector(station_norm: float, wd_norm: float, forecast_dt: pd.Timestamp) -> np.ndarray:
    hour_sin, hour_cos = encode_cyclical(forecast_dt.hour, 24)
    month_sin, month_cos = encode_cyclical(forecast_dt.month, 12)
    dow_sin, dow_cos = encode_cyclical(forecast_dt.dayofweek, 7)
    return np.array(
        [station_norm, wd_norm, hour_sin, hour_cos, month_sin, month_cos, dow_sin, dow_cos],
        dtype=np.float32,
    )


def scale_numeric_rows(rows: list[list[float]], scaler) -> np.ndarray:
    raw = np.asarray(rows, dtype=np.float32)
    return scaler.transform(raw).astype(np.float32)


def build_manual_sequence(
    scaler,
    current_features: dict,
    pm25_history: list[float],
    station_norm: float,
    wd_norm: float,
    forecast_dt: pd.Timestamp,
) -> np.ndarray:
    numeric_rows = []
    for pm25_value in pm25_history:
        numeric_rows.append(
            [
                pm25_value,
                current_features["PM10"],
                current_features["SO2"],
                current_features["NO2"],
                current_features["CO"],
                current_features["O3"],
                current_features["TEMP"],
                current_features["PRES"],
                current_features["DEWP"],
                current_features["RAIN"],
                current_features["WSPM"],
            ]
        )

    scaled_numeric = scale_numeric_rows(numeric_rows, scaler)
    context_vector = build_context_vector(station_norm, wd_norm, forecast_dt)
    repeated_context = np.repeat(context_vector.reshape(1, -1), len(pm25_history), axis=0)
    return np.hstack([scaled_numeric, repeated_context]).reshape(1, len(pm25_history), -1)


def build_dataset_sequence(
    df: pd.DataFrame,
    scaler,
    feature_config: dict,
    station_id: int,
    current_features: dict,
    wd_norm: float,
    forecast_dt: pd.Timestamp,
):
    seq_length = int(feature_config["sequence_length"])
    station_history = df[df["station_encoded"] == station_id].sort_values("datetime").tail(seq_length)

    if len(station_history) < seq_length:
        return None

    sequence = station_history[feature_config["model_feature_columns"]].to_numpy(dtype=np.float32)
    station_norm = float(station_history["station_norm"].iloc[-1])
    scaled_last_step = scale_numeric_rows(
        [
            [
                0.0,
                current_features["PM10"],
                current_features["SO2"],
                current_features["NO2"],
                current_features["CO"],
                current_features["O3"],
                current_features["TEMP"],
                current_features["PRES"],
                current_features["DEWP"],
                current_features["RAIN"],
                current_features["WSPM"],
            ]
        ],
        scaler,
    )[0]

    sequence[-1, 1: len(feature_config["numeric_columns"])] = scaled_last_step[1:]
    sequence[-1, len(feature_config["numeric_columns"]):] = build_context_vector(station_norm, wd_norm, forecast_dt)
    return sequence.reshape(1, seq_length, -1)


def make_prediction(model, sequence: np.ndarray):
    prediction = model.predict(sequence, verbose=0)
    return float(prediction[0][0])


def input_shape_matches(model, feature_config: dict) -> bool:
    expected_features = len(feature_config["model_feature_columns"])
    model_input_shape = model.input_shape
    if isinstance(model_input_shape, list):
        model_input_shape = model_input_shape[0]
    return model_input_shape[-1] == expected_features


local_css("assets/style.css")

scaler, wd_encoder, station_encoder, feature_config = load_resources()
metrics_data = load_metrics()
processed_df = load_processed_data()

st.sidebar.title("SkyGuard AI")
st.sidebar.caption("Spatiotemporal AQI Forecasting")

selection = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Predictions", "Model Insights", "About"],
)

with st.sidebar.expander("System Health", expanded=False):
    checks = {
        "Processed data": DATA_PATH,
        "Feature config": FEATURE_CONFIG_PATH,
        "Scaler": SCALER_PATH,
        "Wind encoder": WD_ENCODER_PATH,
        "Station encoder": STATION_ENCODER_PATH,
        "Metrics": METRICS_PATH,
    }
    for label, path in checks.items():
        st.write(f"{'✅' if os.path.exists(path) else '❌'} {label}: `{path}`")

    st.write("Model files")
    for model_name, path in MODEL_PATHS.items():
        st.write(f"{'✅' if os.path.exists(path) else '❌'} {model_name}: `{path}`")


if selection == "Dashboard":
    st.title("Spatiotemporal Air Quality Dashboard")
    st.write("Explore AQI patterns across stations and over time using real Delhi NCR and Hyderabad station labels for familiarity.")

    if processed_df is None or scaler is None or feature_config is None:
        st.warning("Processed artifacts are missing. Run `python preprocess.py` to refresh the dataset and metadata.")
    else:
        station_names = station_encoder.classes_.tolist()
        station_display_map = build_station_display_map(station_names)
        selected_station_name = st.selectbox(
            "Monitoring station",
            station_names,
            format_func=lambda station_name: station_display_map[station_name],
        )
        selected_station_label = station_display_map[selected_station_name]
        selected_station_id = int(station_encoder.transform([selected_station_name])[0])
        station_df = processed_df[processed_df["station_encoded"] == selected_station_id].sort_values("datetime").copy()

        st.caption("Display labels use real Delhi NCR and Hyderabad monitoring station names, mapped onto the original research dataset.")

        station_df["PM2.5_actual"] = inverse_pm25_series(
            station_df["PM2.5"],
            scaler,
            feature_config["numeric_columns"],
        )
        station_df["AQI"] = station_df["PM2.5_actual"].apply(lambda value: pm25_to_aqi(value)[0])
        station_df["month_name"] = station_df["datetime"].dt.strftime("%b")

        latest_aqi = int(station_df["AQI"].iloc[-1])
        monthly_avg = float(station_df.groupby(station_df["datetime"].dt.month)["AQI"].mean().mean())
        station_peak = float(station_df["PM2.5_actual"].max())

        c1, c2, c3 = st.columns(3)
        c1.metric("Latest AQI", latest_aqi)
        c2.metric("Avg Monthly AQI", f"{monthly_avg:.1f}")
        c3.metric("Peak PM2.5", f"{station_peak:.1f} ug/m3")

        recent_df = station_df.tail(336)
        fig_trend = px.line(
            recent_df,
            x="datetime",
            y=["PM2.5_actual", "AQI"],
            title=f"Recent PM2.5 and AQI trend for {selected_station_label}",
            template="plotly_dark",
        )
        fig_trend.update_layout(xaxis_title="Datetime", yaxis_title="Value")
        st.plotly_chart(fig_trend, use_container_width=True)

        monthly_df = (
            processed_df.assign(
                PM2_5_actual=inverse_pm25_series(
                    processed_df["PM2.5"],
                    scaler,
                    feature_config["numeric_columns"],
                ),
                month_name=processed_df["datetime"].dt.strftime("%b"),
                station_name=station_encoder.inverse_transform(processed_df["station_encoded"].astype(int)),
            )
            .assign(station_label=lambda df: df["station_name"].map(station_display_map))
            .groupby(["station_label", "month_name"], sort=False)["PM2_5_actual"]
            .mean()
            .reset_index()
        )
        fig_heatmap = px.density_heatmap(
            monthly_df,
            x="month_name",
            y="station_label",
            z="PM2_5_actual",
            histfunc="avg",
            title="Average monthly PM2.5 by station",
            color_continuous_scale="YlOrRd",
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        with st.expander("Selected station sample"):
            preview = station_df[["datetime", "PM2.5_actual", "AQI", "TEMP", "WSPM"]].tail(24).copy()
            preview["TEMP"] = preview["TEMP"].round(3)
            preview["WSPM"] = preview["WSPM"].round(3)
            st.dataframe(preview, use_container_width=True)

elif selection == "Predictions":
    st.title("Spatiotemporal AQI Forecaster")
    st.write("Forecast PM2.5 and AQI for a selected station and future time using station-aware sequence inputs.")

    if processed_df is None or scaler is None or feature_config is None:
        st.error("Required artifacts are missing. Run `python preprocess.py` and `python train_models.py` first.")
    else:
        station_names = station_encoder.classes_.tolist()
        station_display_map = build_station_display_map(station_names)
        wind_labels = wd_encoder.classes_.tolist()
        seq_length = int(feature_config["sequence_length"])

        latest_available = processed_df["datetime"].max()
        default_date = latest_available.date() if pd.notna(latest_available) else datetime.today().date()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Location and Time")
            selected_station_name = st.selectbox(
                "Station",
                station_names,
                format_func=lambda station_name: station_display_map[station_name],
            )
            selected_station_label = station_display_map[selected_station_name]
            selected_wd = st.selectbox("Wind direction", wind_labels)
            forecast_date = st.date_input("Forecast date", value=default_date)
            forecast_hour = st.slider("Forecast hour", 0, 23, 12)
            history_mode = st.radio("History source", ["Use station history from dataset", "Manual PM2.5 history"])

        st.caption("Stations are shown with real Delhi NCR and Hyderabad labels; the underlying training data remains the original dataset.")

        with col2:
            st.subheader("Observed Conditions")
            features = {
                "PM10": st.slider("PM10 (ug/m3)", 0.0, 500.0, 60.0),
                "SO2": st.slider("SO2 (ug/m3)", 0.0, 100.0, 12.0),
                "NO2": st.slider("NO2 (ug/m3)", 0.0, 200.0, 35.0),
                "CO": st.slider("CO (ug/m3)", 0.0, 5000.0, 600.0),
                "O3": st.slider("O3 (ug/m3)", 0.0, 300.0, 55.0),
                "TEMP": st.slider("Temperature (C)", -20.0, 45.0, 18.0),
                "PRES": st.number_input("Pressure (hPa)", 950.0, 1050.0, 1010.0),
                "DEWP": st.number_input("Dew point (C)", -30.0, 35.0, 6.0),
                "RAIN": st.slider("Rainfall (mm)", 0.0, 100.0, 0.0),
                "WSPM": st.slider("Wind speed (m/s)", 0.0, 20.0, 2.0),
            }

        history_input = ""
        if history_mode == "Manual PM2.5 history":
            history_input = st.text_area(
                f"PM2.5 history ({seq_length} comma-separated values)",
                value="",
                placeholder=f"Enter exactly {seq_length} PM2.5 values in ug/m3",
                height=110,
            )

        available = {model_name: os.path.exists(path) for model_name, path in MODEL_PATHS.items()}
        model_name = st.selectbox(
            "Model",
            list(MODEL_PATHS.keys()),
            format_func=lambda name: f"{name} {'(missing)' if not available[name] else ''}",
        )

        predict_btn = st.button("Generate Forecast", use_container_width=True)

        if predict_btn:
            if not available[model_name]:
                st.error(f"Model file not found: `{MODEL_PATHS[model_name]}`. Run `python train_models.py`.")
            else:
                model = load_prediction_model(MODEL_PATHS[model_name])
                if model is None:
                    st.error("Unable to load the selected model.")
                elif not input_shape_matches(model, feature_config):
                    st.error(
                        "The selected model was trained on the old feature set. Run `python preprocess.py` and "
                        "`python train_models.py` to rebuild models for spatiotemporal forecasting."
                    )
                else:
                    selected_station_id = int(station_encoder.transform([selected_station_name])[0])
                    wd_id = int(wd_encoder.transform([selected_wd])[0])
                    wd_denominator = max(len(wind_labels) - 1, 1)
                    wd_norm = wd_id / wd_denominator
                    forecast_dt = pd.Timestamp(
                        datetime(
                            forecast_date.year,
                            forecast_date.month,
                            forecast_date.day,
                            forecast_hour,
                        )
                    )

                    if history_mode == "Manual PM2.5 history":
                        history_vals = parse_history_values(history_input, seq_length)
                        if history_vals == "invalid":
                            st.error(f"PM2.5 history must contain exactly {seq_length} numeric values.")
                            history_vals = None
                        elif history_vals is None:
                            st.error("Enter PM2.5 history values or switch to dataset history.")
                        if history_vals is not None:
                            station_norm = float(selected_station_id / max(len(station_names) - 1, 1))
                            sequence = build_manual_sequence(
                                scaler,
                                features,
                                history_vals,
                                station_norm,
                                wd_norm,
                                forecast_dt,
                            )
                        else:
                            sequence = None
                    else:
                        sequence = build_dataset_sequence(
                            processed_df,
                            scaler,
                            feature_config,
                            selected_station_id,
                            features,
                            wd_norm,
                            forecast_dt,
                        )
                        if sequence is None:
                            st.error(f"Not enough history is available for station `{selected_station_label}`.")

                    if sequence is not None:
                        with st.spinner(f"Running {model_name} forecast..."):
                            pred_scaled = make_prediction(model, sequence)

                        pred_pm25 = inverse_pm25_from_scaled(pred_scaled, scaler, feature_config["numeric_columns"])
                        aqi_score, aqi_cat, aqi_color = pm25_to_aqi(pred_pm25)

                        station_df = processed_df[processed_df["station_encoded"] == selected_station_id]
                        recent_station_pm25 = inverse_pm25_series(
                            station_df.tail(seq_length)["PM2.5"],
                            scaler,
                            feature_config["numeric_columns"],
                        )

                        st.success(
                            f"Forecast for {selected_station_label} at {forecast_dt.strftime('%Y-%m-%d %H:%M')}: "
                            f"`{pred_pm25:.2f} ug/m3` PM2.5"
                        )

                        r1, r2, r3 = st.columns(3)
                        r1.metric("AQI Score", aqi_score)
                        r2.metric("AQI Category", aqi_cat)
                        r3.metric("Recent 24h Avg PM2.5", f"{recent_station_pm25.mean():.2f} ug/m3")

                        st.markdown(
                            f"<div style='background:{aqi_color};padding:10px 16px;border-radius:8px;color:#000;"
                            f"font-weight:600;font-size:1.05em;text-align:center;'>{aqi_cat} - AQI {aqi_score}</div>",
                            unsafe_allow_html=True,
                        )

                        comparison_df = pd.DataFrame(
                            {
                                "Step": list(range(1, len(recent_station_pm25) + 1)) + ["Forecast"],
                                "PM2.5": list(recent_station_pm25) + [pred_pm25],
                                "Series": ["History"] * len(recent_station_pm25) + ["Forecast"],
                            }
                        )
                        fig_forecast = px.line(
                            comparison_df,
                            x="Step",
                            y="PM2.5",
                            color="Series",
                            markers=True,
                            title=f"Recent station history vs forecast for {selected_station_label}",
                            template="plotly_dark",
                        )
                        st.plotly_chart(fig_forecast, use_container_width=True)

                        gauge = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=aqi_score,
                                domain={"x": [0, 1], "y": [0, 1]},
                                title={"text": f"AQI Forecast ({model_name})"},
                                gauge={
                                    "axis": {"range": [0, 500]},
                                    "steps": [
                                        {"range": [0, 50], "color": "#00e400"},
                                        {"range": [51, 100], "color": "#ffff00"},
                                        {"range": [101, 150], "color": "#ff7e00"},
                                        {"range": [151, 200], "color": "#ff0000"},
                                        {"range": [201, 300], "color": "#8f3f97"},
                                        {"range": [301, 500], "color": "#7e0023"},
                                    ],
                                    "bar": {"color": "white"},
                                },
                            )
                        )
                        gauge.update_layout(template="plotly_dark")
                        st.plotly_chart(gauge, use_container_width=True)

elif selection == "Model Insights":
    st.title("Model Performance Comparison")
    st.write("Compare spatiotemporal model accuracy after retraining on station-aware sequences.")

    if metrics_data:
        rows = []
        for model_name, values in metrics_data.items():
            rows.append(
                {
                    "Model": model_name,
                    "MAE": values.get("MAE", np.nan),
                    "RMSE": values.get("RMSE", np.nan),
                    "R2": values.get("R2", np.nan),
                }
            )

        metrics_df = pd.DataFrame(rows)
        st.dataframe(metrics_df, use_container_width=True)

        fig = px.bar(
            metrics_df.melt(id_vars="Model", value_vars=["MAE", "RMSE"]),
            x="Model",
            y="value",
            color="variable",
            barmode="group",
            title="MAE and RMSE by model",
            template="plotly_dark",
            color_discrete_sequence=["#7eb0d5", "#fd7f6f"],
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(
            metrics_df,
            x="Model",
            y="R2",
            title="R2 by model",
            template="plotly_dark",
            color="Model",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning(f"Metrics file not found at `{METRICS_PATH}`. Run `python train_models.py`.")

    st.info(
        "If these metrics came from models trained before the refactor, retrain them. The new app expects "
        "station-aware spatiotemporal inputs."
    )

elif selection == "About":
    st.title("About This Project")
    st.markdown(
        """
## Spatiotemporal Air Quality Index Forecasting Using LSTM Networks

This refactored project forecasts PM2.5 and AQI by combining:

- Spatial context from multiple monitoring stations
- Temporal context from hourly sequences and calendar cycles
- Environmental covariates such as PM10, gases, weather, rain, and wind

### What changed
1. Sequences are now built separately for each station instead of mixing locations.
2. The feature set includes station identity, wind direction, hour-of-day, month, and day-of-week context.
3. The prediction workflow now asks for both location and forecast time to match the project title.

### Dataset
Beijing Multi-Site Air-Quality Data Set, 12 stations, hourly observations from 2013 to 2017.

### UI Familiarity Layer
The app shows real Delhi NCR and Hyderabad station labels for easier presentation, but the underlying data and trained models still use the original research dataset.
"""
    )

st.markdown("---")
st.caption("Spatiotemporal AQI Forecasting Using LSTM Networks | Streamlit + TensorFlow")
