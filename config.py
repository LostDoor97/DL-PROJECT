"""
Configuration file for LSTM Stock Prediction Pipeline
"""
import os
from pathlib import Path

# Project Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "evidently_reports"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data Configuration
TICKERS = ['AAPL', 'GOOGL', 'MSFT']  # Only tickers with trained models
START_DATE = '2020-01-01'
END_DATE = None  # Use current date
TRAIN_TEST_SPLIT = 0.8

# Feature Engineering
WINDOW_SIZE = 2500  # Windowed normalization
UNROLLINGS = 50  # Sequence length for LSTM
EMA_SPAN = 10  # EMA smoothing parameter

# Model Hyperparameters
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 64
DENSE_UNITS = 32
DROPOUT_RATE = 0.5
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1

# MLflow Configuration
MLFLOW_TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "stock_lstm_prediction"

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Streamlit Configuration
STREAMLIT_PORT = 8501

# Prediction Configuration
PREDICTION_STEPS = 30  # Multi-step ahead predictions
