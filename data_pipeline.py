"""
Data Pipeline Module for Stock Price Prediction
Handles data fetching, preprocessing, and windowed normalization
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional
from datetime import datetime
import logging
from pathlib import Path

from config import (
    TICKERS, START_DATE, END_DATE, TRAIN_TEST_SPLIT,
    WINDOW_SIZE, UNROLLINGS, RAW_DATA_DIR, PROCESSED_DATA_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataPipeline:
    """Pipeline for fetching and preprocessing stock data"""
    
    def __init__(self, ticker: str, start_date: str = START_DATE, end_date: Optional[str] = END_DATE):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.normalized_data = None
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch OHLCV data from yfinance (or use synthetic data if unavailable)"""
        logger.info(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}")
        
        # First, check if we have cached data
        raw_file = RAW_DATA_DIR / f"{self.ticker}_raw.csv"
        if raw_file.exists():
            logger.info(f"Loading cached data from {raw_file}")
            data = pd.read_csv(raw_file)
            data['Date'] = pd.to_datetime(data['Date'])
            
            # Convert numeric columns to proper types
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Drop rows with NaN in critical columns
            if 'High' in data.columns and 'Low' in data.columns:
                data.dropna(subset=['High', 'Low', 'Close'], inplace=True)
            
            self.data = data
            return data
        
        try:
            # Try fetching with auto_adjust=False to get separate columns
            data = yf.download(
                self.ticker, 
                start=self.start_date, 
                end=self.end_date, 
                progress=False,
                auto_adjust=False,
                actions=False
            )
            
            # If empty, try alternative approach with period
            if data is None or data.empty or len(data) == 0:
                logger.warning(f"No data with date range. Trying period='5y' for {self.ticker}")
                data = yf.download(
                    self.ticker, 
                    period='5y', 
                    progress=False,
                    auto_adjust=False,
                    actions=False
                )
            
            if data is None or data.empty or len(data) == 0:
                logger.warning(f"yfinance failed for {self.ticker}. Generating synthetic data...")
                return self._generate_synthetic_fallback()
            
            logger.info(f"Downloaded {len(data)} rows of data for {self.ticker}")
            logger.info(f"Columns: {data.columns.tolist()}")
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            
            # Handle multi-level columns if present (yfinance sometimes returns these)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Ensure we have required columns
            required_cols = ['High', 'Low', 'Close']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}. Generating synthetic data...")
                return self._generate_synthetic_fallback()
            
            # Convert numeric columns to proper types
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Drop rows with NaN values in required columns
            data.dropna(subset=required_cols, inplace=True)
            
            if len(data) == 0:
                logger.warning(f"No valid data after cleaning for {self.ticker}. Generating synthetic data...")
                return self._generate_synthetic_fallback()
            
            # Save raw data
            data.to_csv(raw_file, index=False)
            logger.info(f"Raw data saved to {raw_file}. Shape: {data.shape}")
            
            self.data = data
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {self.ticker}: {e}")
            logger.warning("Falling back to synthetic data generation...")
            return self._generate_synthetic_fallback()
    
    def _generate_synthetic_fallback(self) -> pd.DataFrame:
        """Generate synthetic data when yfinance is unavailable"""
        from generate_synthetic_data import generate_synthetic_stock_data
        
        logger.info(f"Generating synthetic data for {self.ticker}...")
        data = generate_synthetic_stock_data(self.ticker, self.start_date, self.end_date)
        
        # Save synthetic data
        raw_file = RAW_DATA_DIR / f"{self.ticker}_raw.csv"
        data.to_csv(raw_file, index=False)
        logger.info(f"Synthetic data saved to {raw_file}")
        
        self.data = data
        return data
    
    def compute_mid_price(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Compute mid-price from High and Low"""
        if data is None:
            data = self.data
            
        if data is None:
            raise ValueError("No data available. Run fetch_data() first.")
        
        # Ensure High and Low are numeric
        data['High'] = pd.to_numeric(data['High'], errors='coerce')
        data['Low'] = pd.to_numeric(data['Low'], errors='coerce')
        
        # Drop any rows with NaN values
        data.dropna(subset=['High', 'Low'], inplace=True)
        
        if len(data) == 0:
            raise ValueError(f"No valid numeric data available for {self.ticker}")
        
        # Compute mid-price
        data['Mid'] = (data['High'] + data['Low']) / 2.0
        
        logger.info("Mid-price computed successfully")
        return data
    
    def windowed_normalize(self, data: np.ndarray, window_size: int = WINDOW_SIZE) -> np.ndarray:
        """
        Normalize data using sliding windows
        As per tutorial: normalize in 2500-point windows
        """
        # Ensure data is numpy array and numeric
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float64)
        
        # Check for non-numeric values
        if data.dtype == object:
            logger.warning("Data contains non-numeric values, attempting conversion...")
            data = pd.to_numeric(pd.Series(data), errors='coerce').values
        
        # Remove any NaN values
        if np.any(np.isnan(data)):
            logger.warning(f"Found {np.sum(np.isnan(data))} NaN values, removing them...")
            data = data[~np.isnan(data)]
        
        if len(data) == 0:
            raise ValueError("No valid numeric data to normalize")
        
        normalized_data = []
        
        for i in range(0, len(data), window_size):
            window_end = min(i + window_size, len(data))
            window_data = data[i:window_end].reshape(-1, 1)
            
            # Fit scaler on this window and transform
            scaler = MinMaxScaler(feature_range=(0, 1))
            normalized_window = scaler.fit_transform(window_data)
            normalized_data.extend(normalized_window.flatten())
        
        return np.array(normalized_data)
    
    def create_sequences(self, data: np.ndarray, sequence_length: int = UNROLLINGS) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            data: Normalized price data
            sequence_length: Number of time steps to look back (unrollings)
            
        Returns:
            X: Input sequences of shape (samples, sequence_length, 1)
            y: Target values of shape (samples,)
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        
        X = np.array(X).reshape(-1, sequence_length, 1)
        y = np.array(y)
        
        logger.info(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   split_ratio: float = TRAIN_TEST_SPLIT) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets (80/20)
        """
        split_index = int(len(X) * split_ratio)
        
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        logger.info(f"Train set: X {X_train.shape}, y {y_train.shape}")
        logger.info(f"Test set: X {X_test.shape}, y {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def prepare_data(self, save: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete pipeline: fetch, compute mid-price, normalize, create sequences, split
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Fetch data if not already loaded
        if self.data is None:
            self.fetch_data()
        
        # Compute mid-price
        data = self.compute_mid_price()
        
        # Extract mid-price column
        mid_prices = data['Mid'].values
        
        # Windowed normalization
        logger.info(f"Applying windowed normalization with window size {WINDOW_SIZE}")
        normalized_prices = self.windowed_normalize(np.array(mid_prices))
        
        self.normalized_data = normalized_prices
        
        # Create sequences
        X, y = self.create_sequences(normalized_prices)
        
        # Split into train and test
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Save processed data
        if save:
            processed_file = PROCESSED_DATA_DIR / f"{self.ticker}_processed.npz"
            np.savez(
                processed_file,
                X_train=X_train, X_test=X_test,
                y_train=y_train, y_test=y_test,
                raw_prices=np.array(mid_prices),
                normalized_prices=normalized_prices
            )
            logger.info(f"Processed data saved to {processed_file}")
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def load_processed_data(ticker: str) -> dict:
        """Load previously processed data"""
        processed_file = PROCESSED_DATA_DIR / f"{ticker}_processed.npz"
        
        if not processed_file.exists():
            raise FileNotFoundError(f"Processed data not found: {processed_file}")
        
        data = np.load(processed_file)
        logger.info(f"Loaded processed data from {processed_file}")
        
        return {
            'X_train': data['X_train'],
            'X_test': data['X_test'],
            'y_train': data['y_train'],
            'y_test': data['y_test'],
            'raw_prices': data['raw_prices'],
            'normalized_prices': data['normalized_prices']
        }


def fetch_multiple_tickers(tickers: List[str] = TICKERS) -> dict:
    """Fetch and process data for multiple tickers"""
    results = {}
    
    for ticker in tickers:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing ticker: {ticker}")
            logger.info(f"{'='*50}")
            
            pipeline = StockDataPipeline(ticker)
            X_train, X_test, y_train, y_test = pipeline.prepare_data()
            
            results[ticker] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'pipeline': pipeline
            }
            
        except Exception as e:
            logger.error(f"Failed to process {ticker}: {e}")
            continue
    
    logger.info(f"\nSuccessfully processed {len(results)}/{len(tickers)} tickers")
    return results


if __name__ == "__main__":
    # Test the pipeline with a single ticker
    ticker = "MCC.V"
    pipeline = StockDataPipeline(ticker)
    X_train, X_test, y_train, y_test = pipeline.prepare_data()
    
    print(f"\nData preparation complete for {ticker}:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
