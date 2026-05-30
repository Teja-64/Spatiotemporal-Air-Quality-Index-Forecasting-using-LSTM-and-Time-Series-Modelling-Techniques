# Time-Series Air Quality Index Prediction and Analysis using Multiple Deep Learning Models

This repository implements and compares multiple deep learning architectures to predict **Air Quality Index (AQI)** â€” specifically PM2.5 levels â€” using historical time-series data from 12 monitoring stations in Beijing, China (2013â€“2017).

## ðŸ¤– Implemented Models

| Model | Description |
|---|---|
| **LSTM** | Long Short-Term Memory â€” captures long-range temporal dependencies |
| **BiLSTM** | Bidirectional LSTM â€” processes sequences in both forward and backward directions |
| **BiLSTM+Conv1D** | Conv1D feature extraction followed by Bidirectional LSTM (hybrid architecture) |
| **GRU** | Gated Recurrent Unit â€” computationally efficient alternative to LSTM |

Each model is trained on a **24-hour lookback sliding window** and evaluated with MAE, RMSE, and RÂ² score.

## ðŸ“Š Dataset

- **Source:** Beijing Multi-Site Air-Quality Data Set (Kaggle)
- **Period:** 2013â€“2017 | **Records:** ~420,000 hourly entries
- **Stations:** 12 monitoring sites across Beijing
- **Features:** PM2.5, PM10, SO2, NO2, CO, O3, Temperature, Pressure, Dew Point, Rainfall, Wind Speed

## ðŸš€ How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place Dataset Files
Put all 12 Beijing Air Quality CSV files in the `data/raw/` folder.

### 3. Preprocess Data
```bash
python preprocess.py
```

### 4. Train All Models
```bash
python train_models.py
```
Generates `models/best_lstm_model.keras`, `models/best_bilstm_model.keras`, `models/best_gru_model.keras`, `models/best_bilstm_conv1d_model.keras`, and `models/metrics.json`.

### 5. Launch the Web App
```bash
streamlit run app.py
```

## ðŸŒ App Features

- **Dashboard** â€” Real historical pollutant trends and distributions from the dataset
- **Predictions** â€” Select any of 4 models, input environmental parameters â†’ get PM2.5 prediction + **AQI score & category** (US EPA standard)
- **Model Insights** â€” Side-by-side MAE / RMSE / RÂ² comparison from actual training results
- **About** â€” Full project methodology and architecture overview

## ðŸ› ï¸ Tech Stack

- **Deep Learning:** TensorFlow 2.x / Keras
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Web App:** Streamlit + Plotly
- **AQI Standard:** US EPA PM2.5 breakpoints

## ðŸ““ Jupyter Notebooks

Individual exploratory notebooks are also provided:
- `notebooks/LSTM.ipynb`
- `notebooks/BiLSTM.ipynb`
- `notebooks/BiLSTM_Conv1D.ipynb`
- `notebooks/GRU.ipynb`
- `notebooks/LSTM_BiLSTM_BiLSTMCNN_.ipynb` â€” consolidated comparison notebook
- `notebooks/Data_Pre_Processing_.ipynb`

---
**B.Tech Major Project â€” SkyGuard AI Â© 2026**

## Project Structure

```text
.
|-- app.py
|-- preprocess.py
|-- train_models.py
|-- assets/
|-- artifacts/
|   |-- scaler.joblib
|   |-- station_encoder.joblib
|   `-- wd_encoder.joblib
|-- data/
|   |-- raw/
|   `-- processed/
|       `-- processed_data.csv
|-- docs/
|   |-- PROJECT_REPORT.md
|   `-- RUN.md
|-- models/
|   |-- best_lstm_model.keras
|   |-- best_bilstm_model.keras
|   |-- best_gru_model.keras
|   |-- best_bilstm_conv1d_model.keras
|   `-- metrics.json
`-- notebooks/
```
