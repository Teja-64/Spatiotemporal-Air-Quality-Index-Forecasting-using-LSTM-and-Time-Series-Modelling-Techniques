import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
SCALER_PATH = "artifacts/scaler.joblib"
WD_ENCODER_PATH = "artifacts/wd_encoder.joblib"
STATION_ENCODER_PATH = "artifacts/station_encoder.joblib"
FEATURE_CONFIG_PATH = "artifacts/feature_config.json"

NUMERIC_COLUMNS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
TIME_FEATURE_COLUMNS = ["hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos"]
STATIC_CONTEXT_COLUMNS = ["station_norm", "wd_norm"]
MODEL_FEATURE_COLUMNS = NUMERIC_COLUMNS + STATIC_CONTEXT_COLUMNS + TIME_FEATURE_COLUMNS


def encode_cyclical(series: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    radians = 2 * np.pi * series.astype(float) / period
    return np.sin(radians), np.cos(radians)


def fill_station_gaps(df: pd.DataFrame) -> pd.DataFrame:
    filled_frames = []
    for _, station_df in df.groupby("station", sort=False):
        station_df = station_df.sort_values(["year", "month", "day", "hour"]).copy()
        station_df[NUMERIC_COLUMNS] = station_df[NUMERIC_COLUMNS].interpolate(method="linear", limit_direction="both")
        station_df[NUMERIC_COLUMNS] = station_df[NUMERIC_COLUMNS].bfill().ffill()
        filled_frames.append(station_df)
    return pd.concat(filled_frames, ignore_index=True)


def build_feature_config(sequence_length: int, station_encoder: LabelEncoder, wd_encoder: LabelEncoder) -> dict:
    return {
        "sequence_length": sequence_length,
        "numeric_columns": NUMERIC_COLUMNS,
        "time_feature_columns": TIME_FEATURE_COLUMNS,
        "context_columns": STATIC_CONTEXT_COLUMNS,
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "station_labels": station_encoder.classes_.tolist(),
        "wd_labels": wd_encoder.classes_.tolist(),
    }


def preprocess_data(data_dir: str = RAW_DATA_DIR, output_file: str = PROCESSED_DATA_PATH, seq_length: int = 24):
    all_files = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir) if file_name.endswith(".csv")]

    if not all_files:
        print(f"No CSV files found in {data_dir}. Please place the dataset files there.")
        return None

    data_frames = [pd.read_csv(file_name) for file_name in sorted(all_files)]
    df = pd.concat(data_frames, axis=0, ignore_index=True)

    if "No" in df.columns:
        df = df.drop(columns=["No"])

    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()

    df = fill_station_gaps(df)

    wd_encoder = LabelEncoder()
    df["wd_encoded"] = wd_encoder.fit_transform(df["wd"].astype(str))

    station_encoder = LabelEncoder()
    df["station_encoded"] = station_encoder.fit_transform(df["station"].astype(str))

    station_denominator = max(len(station_encoder.classes_) - 1, 1)
    wd_denominator = max(len(wd_encoder.classes_) - 1, 1)
    df["station_norm"] = df["station_encoded"] / station_denominator
    df["wd_norm"] = df["wd_encoded"] / wd_denominator

    df["hour_sin"], df["hour_cos"] = encode_cyclical(df["hour"], 24)
    df["month_sin"], df["month_cos"] = encode_cyclical(df["month"], 12)
    day_of_week = df["datetime"].dt.dayofweek
    df["dow_sin"], df["dow_cos"] = encode_cyclical(day_of_week, 7)

    scaler = MinMaxScaler()
    df[NUMERIC_COLUMNS] = scaler.fit_transform(df[NUMERIC_COLUMNS])

    df = df.sort_values(["station_encoded", "datetime"]).reset_index(drop=True)

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(wd_encoder, WD_ENCODER_PATH)
    joblib.dump(station_encoder, STATION_ENCODER_PATH)

    feature_config = build_feature_config(seq_length, station_encoder, wd_encoder)
    with open(FEATURE_CONFIG_PATH, "w", encoding="utf-8") as config_file:
        json.dump(feature_config, config_file, indent=2)

    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")
    return df


if __name__ == "__main__":
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    preprocess_data()
