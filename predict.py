"""
Prediction Module with Multi-Step Ahead Forecasting and EMA Smoothing
"""
import numpy as np
import pandas as pd
from keras.models import load_model
from typing import Optional, Tuple
import logging
from pathlib import Path

from config import PREDICTION_STEPS, EMA_SPAN, UNROLLINGS, MODELS_DIR
from data_pipeline import StockDataPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockPredictor:
    """Multi-step ahead stock price predictor with EMA smoothing"""
    
    def __init__(self, model_path: str, ticker: str):
        self.model_path = model_path
        self.ticker = ticker
        self.model = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load trained model"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = load_model(self.model_path)
        logger.info(f"Model loaded from {self.model_path}")
        
        # Initialize pipeline for data preprocessing
        self.pipeline = StockDataPipeline(self.ticker)
    
    def prepare_input_sequence(self, data: np.ndarray, sequence_length: int = UNROLLINGS) -> np.ndarray:
        """
        Prepare input sequence for prediction
        
        Args:
            data: Normalized price data
            sequence_length: Number of time steps
            
        Returns:
            Input sequence of shape (1, sequence_length, 1)
        """
        if len(data) < sequence_length:
            raise ValueError(f"Not enough data. Need at least {sequence_length} points, got {len(data)}")
        
        # Get last sequence_length points
        sequence = data[-sequence_length:]
        return sequence.reshape(1, sequence_length, 1)
    
    def predict_single_step(self, input_sequence: np.ndarray) -> float:
        """
        Predict next single step
        
        Args:
            input_sequence: Input of shape (1, sequence_length, 1)
            
        Returns:
            Predicted value (normalized)
        """
        prediction = self.model.predict(input_sequence, verbose='silent')
        return prediction[0, 0]
    
    def predict_multi_step(self, initial_sequence: np.ndarray, 
                          steps: int = PREDICTION_STEPS) -> np.ndarray:
        """
        Multi-step ahead prediction using recursive approach
        Uses prior predictions as input for subsequent predictions
        
        Args:
            initial_sequence: Starting sequence of normalized prices
            steps: Number of steps to predict ahead
            
        Returns:
            Array of predicted values (normalized)
        """
        logger.info(f"Predicting {steps} steps ahead...")
        
        predictions = []
        current_sequence = initial_sequence.copy()
        
        for step in range(steps):
            # Prepare input
            input_seq = self.prepare_input_sequence(current_sequence)
            
            # Predict next step
            next_pred = self.predict_single_step(input_seq)
            predictions.append(next_pred)
            
            # Update sequence with prediction
            current_sequence = np.append(current_sequence, next_pred)
        
        predictions = np.array(predictions)
        logger.info(f"Multi-step prediction completed: {len(predictions)} steps")
        
        return predictions
    
    @staticmethod
    def apply_ema_smoothing(data: np.ndarray, span: int = EMA_SPAN) -> np.ndarray:
        """
        Apply Exponential Moving Average (EMA) smoothing
        
        Args:
            data: Input data array
            span: Span for EMA calculation
            
        Returns:
            Smoothed data array
        """
        df = pd.DataFrame(data, columns=['value'])
        ema = df['value'].ewm(span=span, adjust=False).mean()
        
        logger.info(f"Applied EMA smoothing with span={span}")
        return np.array(ema.values)
    
    def predict_and_smooth(self, initial_sequence: np.ndarray, 
                          steps: int = PREDICTION_STEPS,
                          apply_smoothing: bool = True,
                          ema_span: int = EMA_SPAN) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict multi-step ahead and apply EMA smoothing
        
        Args:
            initial_sequence: Starting sequence
            steps: Number of steps to predict
            apply_smoothing: Whether to apply EMA smoothing
            ema_span: Span for EMA
            
        Returns:
            Tuple of (raw_predictions, smoothed_predictions)
        """
        # Get raw predictions
        raw_predictions = self.predict_multi_step(initial_sequence, steps)
        
        # Apply smoothing if requested
        if apply_smoothing:
            smoothed_predictions = self.apply_ema_smoothing(raw_predictions, span=ema_span)
        else:
            smoothed_predictions = raw_predictions
        
        return raw_predictions, smoothed_predictions
    
    def predict_from_ticker(self, steps: int = PREDICTION_STEPS,
                           apply_smoothing: bool = True) -> dict:
        """
        End-to-end prediction from ticker
        Fetches latest data, makes predictions, and applies smoothing
        
        Returns:
            Dictionary with predictions and metadata
        """
        logger.info(f"Generating predictions for {self.ticker}...")
        
        # Fetch and prepare data
        self.pipeline.fetch_data()
        data = self.pipeline.compute_mid_price()
        
        # Get normalized prices
        mid_prices = data['Mid'].values
        normalized_prices = self.pipeline.windowed_normalize(np.array(mid_prices))
        
        # Predict
        raw_predictions, smoothed_predictions = self.predict_and_smooth(
            normalized_prices,
            steps=steps,
            apply_smoothing=apply_smoothing
        )
        
        result = {
            'ticker': self.ticker,
            'prediction_steps': steps,
            'raw_predictions': raw_predictions.tolist(),
            'smoothed_predictions': smoothed_predictions.tolist(),
            'last_actual_value': float(normalized_prices[-1]),
            'latest_date': str(data['Date'].iloc[-1]),
            'model_path': self.model_path
        }
        
        logger.info(f"Prediction completed for {self.ticker}")
        return result
    
    def evaluate_predictions(self, X_test: np.ndarray, y_test: np.ndarray,
                           steps: int = 1) -> dict:
        """
        Evaluate prediction accuracy on test set
        
        Args:
            X_test: Test sequences
            y_test: True values
            steps: Number of steps ahead to predict (1 for single-step)
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = []
        
        if steps == 1:
            # Single-step predictions
            predictions = self.model.predict(X_test, verbose='silent').flatten()
        else:
            # Multi-step predictions
            for i in range(len(X_test)):
                initial_seq = X_test[i].flatten()
                multi_pred = self.predict_multi_step(initial_seq, steps=steps)
                predictions.append(multi_pred[-1])  # Use last prediction
            predictions = np.array(predictions)
        
        # Calculate metrics
        mse = np.mean((y_test - predictions) ** 2)
        mae = np.mean(np.abs(y_test - predictions))
        rmse = np.sqrt(mse)
        
        # Direction accuracy
        if len(y_test) > 1:
            true_directions = np.diff(y_test) > 0
            pred_directions = np.diff(predictions) > 0
            direction_accuracy = np.mean(true_directions == pred_directions) * 100
        else:
            direction_accuracy = 0.0
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'direction_accuracy': float(direction_accuracy),
            'prediction_steps': steps
        }
        
        logger.info(f"Evaluation metrics (steps={steps}):")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.6f}")
        
        return metrics


def find_latest_model(ticker: str, models_dir: Path = MODELS_DIR) -> Optional[str]:
    """Find the most recent model file for a ticker"""
    pattern = f"{ticker}_lstm_*.h5"
    model_files = list(models_dir.glob(pattern))
    
    if not model_files:
        logger.warning(f"No model found for {ticker}")
        return None
    
    # Sort by modification time and get the latest
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Found latest model: {latest_model}")
    
    return str(latest_model)


if __name__ == "__main__":
    # Example usage
    ticker = "MCC.V"
    
    # Find latest model
    model_path = find_latest_model(ticker)
    
    if model_path:
        # Create predictor
        predictor = StockPredictor(model_path, ticker)
        
        # Generate predictions
        result = predictor.predict_from_ticker(steps=30, apply_smoothing=True)
        
        print(f"\nPredictions for {ticker}:")
        print(f"Last actual value: {result['last_actual_value']:.6f}")
        print(f"Predicted next {len(result['smoothed_predictions'])} steps")
        print(f"Smoothed predictions (first 5): {result['smoothed_predictions'][:5]}")
    else:
        print(f"No model found for {ticker}. Train a model first.")
