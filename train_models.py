import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Conv1D, Dense, Dropout, Input, MaxPooling1D
from tensorflow.keras.models import Sequential


PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
FEATURE_CONFIG_PATH = "artifacts/feature_config.json"
MODELS_DIR = "models"
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")


def load_feature_config():
    if not os.path.exists(FEATURE_CONFIG_PATH):
        raise FileNotFoundError(f"Feature config not found at {FEATURE_CONFIG_PATH}. Run preprocess.py first.")
    with open(FEATURE_CONFIG_PATH, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def create_station_sequences(df: pd.DataFrame, feature_columns: list[str], seq_length: int):
    x_sequences = []
    y_targets = []
    metadata = []

    for station_id, station_df in df.groupby("station_encoded", sort=False):
        station_df = station_df.sort_values("datetime").reset_index(drop=True)
        feature_values = station_df[feature_columns].to_numpy(dtype=np.float32)
        target_values = station_df["PM2.5"].to_numpy(dtype=np.float32)
        datetimes = pd.to_datetime(station_df["datetime"])

        if len(station_df) <= seq_length:
            continue

        for start_idx in range(len(station_df) - seq_length):
            end_idx = start_idx + seq_length
            x_sequences.append(feature_values[start_idx:end_idx])
            y_targets.append(target_values[end_idx])
            metadata.append((datetimes.iloc[end_idx], int(station_id)))

    if not x_sequences:
        raise ValueError("No training sequences were created. Check the processed dataset and sequence length.")

    x = np.asarray(x_sequences, dtype=np.float32)
    y = np.asarray(y_targets, dtype=np.float32)
    meta_df = pd.DataFrame(metadata, columns=["target_datetime", "station_encoded"])

    ordering = np.argsort(meta_df["target_datetime"].to_numpy(dtype="datetime64[ns]"))
    return x[ordering], y[ordering], meta_df.iloc[ordering].reset_index(drop=True)


def split_by_time(x: np.ndarray, y: np.ndarray, train_ratio: float = 0.8):
    split_index = max(int(len(x) * train_ratio), 1)
    split_index = min(split_index, len(x) - 1)
    return x[:split_index], x[split_index:], y[:split_index], y[split_index:]


def evaluate_model(model, x_test: np.ndarray, y_test: np.ndarray):
    predictions = model.predict(x_test, verbose=0).flatten()
    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def fit_and_save(model_name: str, model: Sequential, x_train, y_train, x_test, y_test, file_name: str):
    callbacks = [EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        callbacks=callbacks,
        shuffle=False,
        verbose=1,
    )
    model.save(os.path.join(MODELS_DIR, file_name))
    metrics = evaluate_model(model, x_test, y_test)
    print(f"{model_name} metrics: {metrics}")
    return metrics


def train_and_save_models(processed_data_path: str = PROCESSED_DATA_PATH):
    if not os.path.exists(processed_data_path):
        print(f"Processed data not found at {processed_data_path}. Run preprocess.py first.")
        return

    feature_config = load_feature_config()
    feature_columns = feature_config["model_feature_columns"]
    seq_length = int(feature_config["sequence_length"])

    df = pd.read_csv(processed_data_path, parse_dates=["datetime"])
    required_columns = set(feature_columns + ["station_encoded", "PM2.5", "datetime"])
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Processed dataset is missing required columns: {missing_columns}")

    x, y, _ = create_station_sequences(df, feature_columns, seq_length)
    x_train, x_test, y_train, y_test = split_by_time(x, y)

    os.makedirs(MODELS_DIR, exist_ok=True)
    input_shape = (seq_length, x.shape[2])

    all_metrics = {}

    print("Training LSTM model on spatiotemporal feature set...")
    lstm_model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(64, activation="tanh"),
            Dropout(0.2),
            Dense(1),
        ]
    )
    all_metrics["LSTM"] = fit_and_save(
        "LSTM",
        lstm_model,
        x_train,
        y_train,
        x_test,
        y_test,
        "best_lstm_model.keras",
    )

    print("Training BiLSTM model on spatiotemporal feature set...")
    bilstm_model = Sequential(
        [
            Input(shape=input_shape),
            Bidirectional(LSTM(64, activation="tanh")),
            Dropout(0.2),
            Dense(1),
        ]
    )
    all_metrics["BiLSTM"] = fit_and_save(
        "BiLSTM",
        bilstm_model,
        x_train,
        y_train,
        x_test,
        y_test,
        "best_bilstm_model.keras",
    )

    print("Training GRU model on spatiotemporal feature set...")
    gru_model = Sequential(
        [
            Input(shape=input_shape),
            GRU(64, activation="tanh"),
            Dropout(0.2),
            Dense(1),
        ]
    )
    all_metrics["GRU"] = fit_and_save(
        "GRU",
        gru_model,
        x_train,
        y_train,
        x_test,
        y_test,
        "best_gru_model.keras",
    )

    print("Training BiLSTM+Conv1D model on spatiotemporal feature set...")
    bilstm_conv_model = Sequential(
        [
            Input(shape=input_shape),
            Conv1D(filters=64, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Bidirectional(LSTM(64, activation="tanh")),
            Dropout(0.2),
            Dense(1),
        ]
    )
    all_metrics["BiLSTM+Conv1D"] = fit_and_save(
        "BiLSTM+Conv1D",
        bilstm_conv_model,
        x_train,
        y_train,
        x_test,
        y_test,
        "best_bilstm_conv1d_model.keras",
    )

    with open(METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        json.dump(all_metrics, metrics_file, indent=2)
    print(f"All models trained and saved. Metrics written to {METRICS_PATH}.")


if __name__ == "__main__":
    train_and_save_models()
