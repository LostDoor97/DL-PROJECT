"""
Training Pipeline with MLflow Integration
Logs hyperparameters, metrics, and models to MLflow
"""
import numpy as np
import mlflow
import mlflow.keras
from datetime import datetime
import logging
from pathlib import Path
import json
from typing import Optional

from data_pipeline import StockDataPipeline
from model import create_model
from config import (
    UNROLLINGS, BATCH_SIZE, EPOCHS, VALIDATION_SPLIT,
    MLFLOW_TRACKING_URI, EXPERIMENT_NAME, MODELS_DIR,
    LEARNING_RATE, TICKERS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowTrainer:
    """Training pipeline with MLflow tracking"""
    
    def __init__(self, ticker: str, experiment_name: str = EXPERIMENT_NAME):
        self.ticker = ticker
        self.experiment_name = experiment_name
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created new experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.experiment_name}")
            
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.error(f"Error setting up MLflow: {e}")
            raise
    
    def train_model(self, run_name: Optional[str] = None,
                   batch_size: int = BATCH_SIZE,
                   epochs: int = EPOCHS,
                   learning_rate: float = LEARNING_RATE,
                   unrollings: int = UNROLLINGS) -> dict:
        """
        Train LSTM model with MLflow tracking
        
        Returns:
            Dictionary with metrics and model path
        """
        if run_name is None:
            run_name = f"{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name) as run:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting MLflow run: {run_name}")
            logger.info(f"Run ID: {run.info.run_id}")
            logger.info(f"{'='*60}\n")
            
            # Log parameters
            params = {
                'ticker': self.ticker,
                'unrollings': unrollings,
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'validation_split': VALIDATION_SPLIT,
                'model_type': 'LSTM',
                'lstm_units_1': 64,
                'lstm_units_2': 64,
                'dense_units': 32,
                'dropout_rate': 0.5
            }
            mlflow.log_params(params)
            logger.info(f"Logged parameters: {params}")
            
            # Prepare data
            logger.info("Preparing data...")
            pipeline = StockDataPipeline(self.ticker)
            X_train, X_test, y_train, y_test = pipeline.prepare_data()
            
            # Log data metrics
            mlflow.log_metric("train_samples", len(X_train))
            mlflow.log_metric("test_samples", len(X_test))
            
            # Create and train model
            logger.info("Creating model...")
            model = create_model(sequence_length=unrollings, learning_rate=learning_rate)
            
            logger.info("Training model...")
            history = model.train(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=VALIDATION_SPLIT,
                verbose=1
            )
            
            # Log training metrics
            for epoch in range(len(history.history['loss'])):
                mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
                mlflow.log_metric("train_mae", history.history['mae'][epoch], step=epoch)
                mlflow.log_metric("train_mse", history.history['mse'][epoch], step=epoch)
                
                if 'val_loss' in history.history:
                    mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
                    mlflow.log_metric("val_mae", history.history['val_mae'][epoch], step=epoch)
                    mlflow.log_metric("val_mse", history.history['val_mse'][epoch], step=epoch)
            
            # Evaluate model
            logger.info("Evaluating model...")
            metrics = model.evaluate(X_test, y_test)
            
            # Log test metrics
            mlflow.log_metrics({
                'test_loss': metrics['loss'],
                'test_mae': metrics['mae'],
                'test_mse': metrics['mse'],
                'test_rmse': metrics['rmse'],
                'direction_accuracy': metrics['direction_accuracy']
            })
            
            logger.info(f"Test Metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.6f}")
            
            # Save model
            model_filename = f"{self.ticker}_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
            model_path = MODELS_DIR / model_filename
            model.save(str(model_path))
            
            # Log model to MLflow
            mlflow.keras.log_model(model.model, "model")
            
            # Log model file as artifact
            mlflow.log_artifact(str(model_path))
            
            # Save and log configuration
            config = {
                'ticker': self.ticker,
                'hyperparameters': params,
                'metrics': metrics,
                'model_path': str(model_path),
                'run_id': run.info.run_id
            }
            
            config_path = MODELS_DIR / f"{self.ticker}_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            mlflow.log_artifact(str(config_path))
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Training completed successfully!")
            logger.info(f"Model saved to: {model_path}")
            logger.info(f"Direction Accuracy: {metrics['direction_accuracy']:.2f}%")
            logger.info(f"{'='*60}\n")
            
            return {
                'run_id': run.info.run_id,
                'metrics': metrics,
                'model_path': str(model_path),
                'params': params
            }


def train_multiple_tickers(tickers: Optional[list] = None, **kwargs):
    """Train models for multiple tickers"""
    if tickers is None:
        tickers = TICKERS[:3]  # Start with first 3 tickers
    
    results = {}
    
    for ticker in tickers:
        try:
            logger.info(f"\n\n{'#'*70}")
            logger.info(f"# Training model for {ticker}")
            logger.info(f"{'#'*70}\n")
            
            trainer = MLflowTrainer(ticker)
            result = trainer.train_model(**kwargs)
            results[ticker] = result
            
        except Exception as e:
            logger.error(f"Failed to train model for {ticker}: {e}")
            continue
    
    # Summary
    logger.info(f"\n\n{'='*70}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*70}")
    
    for ticker, result in results.items():
        logger.info(f"\n{ticker}:")
        logger.info(f"  Run ID: {result['run_id']}")
        logger.info(f"  Direction Accuracy: {result['metrics']['direction_accuracy']:.2f}%")
        logger.info(f"  RMSE: {result['metrics']['rmse']:.6f}")
        logger.info(f"  Model: {result['model_path']}")
    
    logger.info(f"\n{'='*70}\n")
    
    return results


if __name__ == "__main__":
    # Train a single model
    ticker = "MCC.V"
    trainer = MLflowTrainer(ticker)
    result = trainer.train_model(epochs=50)
    
    print(f"\nTraining completed!")
    print(f"Direction Accuracy: {result['metrics']['direction_accuracy']:.2f}%")
    print(f"Model saved to: {result['model_path']}")
