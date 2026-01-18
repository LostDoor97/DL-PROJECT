"""
Baseline ARIMA Model for MLflow Comparison
"""
import numpy as np
import pandas as pd
import mlflow
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from datetime import datetime
import logging
from typing import Optional, Tuple

from data_pipeline import StockDataPipeline
from config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME, PREDICTION_STEPS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ARIMABaseline:
    """ARIMA baseline model for comparison with LSTM"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.model = None
        self.fitted_model = None
        self.order = None
    
    def auto_find_order(self, data: np.ndarray, max_p: int = 5, 
                       max_d: int = 2, max_q: int = 5) -> Tuple[int, int, int]:
        """
        Automatically find best ARIMA order using auto_arima
        
        Args:
            data: Time series data
            max_p: Maximum AR order
            max_d: Maximum differencing order
            max_q: Maximum MA order
            
        Returns:
            Tuple of (p, d, q)
        """
        logger.info("Finding optimal ARIMA order...")
        
        model = auto_arima(
            data,
            start_p=0, start_q=0,
            max_p=max_p, max_d=max_d, max_q=max_q,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )
        
        self.order = model.order
        logger.info(f"Optimal ARIMA order: {self.order}")
        
        return self.order
    
    def train(self, train_data: np.ndarray, order: Optional[Tuple[int, int, int]] = None):
        """
        Train ARIMA model
        
        Args:
            train_data: Training time series data
            order: ARIMA order (p, d, q). If None, will auto-detect
        """
        if order is None:
            order = self.auto_find_order(train_data)
        else:
            self.order = order
        
        logger.info(f"Training ARIMA{order} model...")
        
        self.model = ARIMA(train_data, order=order)
        self.fitted_model = self.model.fit()
        
        logger.info("ARIMA model training completed")
        logger.info(f"AIC: {self.fitted_model.aic:.2f}")
        logger.info(f"BIC: {self.fitted_model.bic:.2f}")
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Make multi-step ahead predictions
        
        Args:
            steps: Number of steps to predict ahead
            
        Returns:
            Array of predictions
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return np.array(forecast)
    
    def evaluate(self, test_data: np.ndarray) -> dict:
        """
        Evaluate model on test data
        
        Args:
            test_data: Test time series data
            
        Returns:
            Dictionary of metrics
        """
        # Predict for test set length
        predictions = self.predict(steps=len(test_data))
        
        # Calculate metrics
        mse = np.mean((test_data - predictions) ** 2)
        mae = np.mean(np.abs(test_data - predictions))
        rmse = np.sqrt(mse)
        
        # Direction accuracy
        if len(test_data) > 1:
            true_directions = np.diff(test_data) > 0
            pred_directions = np.diff(predictions) > 0
            direction_accuracy = np.mean(true_directions == pred_directions) * 100
        else:
            direction_accuracy = 0.0
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'direction_accuracy': float(direction_accuracy),
            'aic': float(self.fitted_model.aic),
            'bic': float(self.fitted_model.bic)
        }
        
        logger.info("ARIMA Evaluation Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.6f}")
        
        return metrics


class ARIMAMLflowTrainer:
    """Train ARIMA model with MLflow tracking"""
    
    def __init__(self, ticker: str, experiment_name: str = EXPERIMENT_NAME):
        self.ticker = ticker
        self.experiment_name = experiment_name
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.experiment_name)
    
    def train_and_log(self, order: Optional[Tuple[int, int, int]] = None) -> dict:
        """
        Train ARIMA model and log to MLflow
        
        Args:
            order: ARIMA order (p, d, q). If None, will auto-detect
            
        Returns:
            Dictionary with metrics and run info
        """
        run_name = f"{self.ticker}_ARIMA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name) as run:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting ARIMA MLflow run: {run_name}")
            logger.info(f"Run ID: {run.info.run_id}")
            logger.info(f"{'='*60}\n")
            
            # Prepare data
            logger.info("Preparing data...")
            pipeline = StockDataPipeline(self.ticker)
            pipeline.fetch_data()
            data = pipeline.compute_mid_price()
            
            # Use normalized mid-prices
            mid_prices = data['Mid'].values
            normalized_prices = pipeline.windowed_normalize(np.array(mid_prices))
            
            # Split data (80/20)
            split_idx = int(len(normalized_prices) * 0.8)
            train_data = normalized_prices[:split_idx]
            test_data = normalized_prices[split_idx:]
            
            # Create and train model
            arima_model = ARIMABaseline(self.ticker)
            arima_model.train(train_data, order=order)
            
            # Log parameters
            params = {
                'ticker': self.ticker,
                'model_type': 'ARIMA',
                'p': arima_model.order[0],
                'd': arima_model.order[1],
                'q': arima_model.order[2],
                'order': str(arima_model.order),
                'train_samples': len(train_data),
                'test_samples': len(test_data)
            }
            mlflow.log_params(params)
            
            # Evaluate
            metrics = arima_model.evaluate(test_data)
            
            # Log metrics
            mlflow.log_metrics({
                'test_mse': metrics['mse'],
                'test_mae': metrics['mae'],
                'test_rmse': metrics['rmse'],
                'direction_accuracy': metrics['direction_accuracy'],
                'aic': metrics['aic'],
                'bic': metrics['bic']
            })
            
            logger.info(f"\n{'='*60}")
            logger.info(f"ARIMA Training completed!")
            logger.info(f"Direction Accuracy: {metrics['direction_accuracy']:.2f}%")
            logger.info(f"RMSE: {metrics['rmse']:.6f}")
            logger.info(f"{'='*60}\n")
            
            return {
                'run_id': run.info.run_id,
                'metrics': metrics,
                'params': params,
                'model': arima_model
            }


def compare_models(ticker: str):
    """
    Train both LSTM and ARIMA models and compare in MLflow
    """
    from train import MLflowTrainer
    
    logger.info(f"\n{'#'*70}")
    logger.info(f"# Model Comparison for {ticker}: LSTM vs ARIMA")
    logger.info(f"{'#'*70}\n")
    
    results = {}
    
    # Train LSTM
    try:
        logger.info("Training LSTM model...")
        lstm_trainer = MLflowTrainer(ticker)
        lstm_result = lstm_trainer.train_model(epochs=30)
        results['LSTM'] = lstm_result
    except Exception as e:
        logger.error(f"LSTM training failed: {e}")
    
    # Train ARIMA
    try:
        logger.info("\nTraining ARIMA model...")
        arima_trainer = ARIMAMLflowTrainer(ticker)
        arima_result = arima_trainer.train_and_log()
        results['ARIMA'] = arima_result
    except Exception as e:
        logger.error(f"ARIMA training failed: {e}")
    
    # Compare results
    if len(results) == 2:
        logger.info(f"\n{'='*70}")
        logger.info("MODEL COMPARISON")
        logger.info(f"{'='*70}")
        
        for model_name, result in results.items():
            logger.info(f"\n{model_name}:")
            logger.info(f"  Direction Accuracy: {result['metrics']['direction_accuracy']:.2f}%")
            logger.info(f"  RMSE: {result['metrics']['rmse']:.6f}")
            logger.info(f"  Run ID: {result['run_id']}")
        
        # Determine winner
        lstm_acc = results['LSTM']['metrics']['direction_accuracy']
        arima_acc = results['ARIMA']['metrics']['direction_accuracy']
        
        winner = 'LSTM' if lstm_acc > arima_acc else 'ARIMA'
        logger.info(f"\nBest Model: {winner}")
        logger.info(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    # Compare LSTM vs ARIMA for a ticker
    ticker = "MCC.V"
    results = compare_models(ticker)
