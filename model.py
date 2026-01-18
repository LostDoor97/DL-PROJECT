"""
LSTM Model Architecture
Stacked LSTM with Dropout for Stock Price Prediction
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import logging
from typing import Tuple, Optional

from config import (
    LSTM_UNITS_1, LSTM_UNITS_2, DENSE_UNITS, DROPOUT_RATE,
    LEARNING_RATE, MODELS_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMStockModel:
    """
    Stacked LSTM Model for Stock Price Prediction
    Architecture based on DATACAMP tutorial:
    - LSTM(64, return_sequences=True)
    - LSTM(64)
    - Dense(32)
    - Dropout(0.5)
    - Dense(1)
    """
    
    def __init__(self, sequence_length: int, learning_rate: float = LEARNING_RATE):
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """Build the stacked LSTM model"""
        logger.info("Building LSTM model architecture...")
        
        self.model = Sequential([
            # First LSTM layer with return_sequences=True
            LSTM(
                units=LSTM_UNITS_1,
                return_sequences=True,
                input_shape=(self.sequence_length, 1),
                name='lstm_1'
            ),
            
            # Second LSTM layer
            LSTM(
                units=LSTM_UNITS_2,
                return_sequences=False,
                name='lstm_2'
            ),
            
            # Dense layer
            Dense(
                units=DENSE_UNITS,
                activation='relu',
                name='dense_1'
            ),
            
            # Dropout for regularization
            Dropout(
                rate=DROPOUT_RATE,
                name='dropout'
            ),
            
            # Output layer
            Dense(
                units=1,
                activation='linear',
                name='output'
            )
        ])
        
        # Compile model with MSE loss and Adam optimizer
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        logger.info("Model architecture built successfully")
        logger.info(f"\nModel Summary:")
        self.model.summary(print_fn=logger.info)
    
    def get_callbacks(self, model_path: Optional[str] = None) -> list:
        """Get training callbacks"""
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint
        if model_path is None:
            model_path = str(MODELS_DIR / "best_model.h5")
        
        checkpoint = ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        return callbacks
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              batch_size: int = 32, epochs: int = 50,
              validation_split: float = 0.1, verbose: int = 1) -> keras.callbacks.History:
        """
        Train the LSTM model
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            batch_size: Batch size for training
            epochs: Number of training epochs
            validation_split: Fraction of training data to use for validation
            verbose: Verbosity mode
            
        Returns:
            Training history
        """
        logger.info(f"Starting training with {len(X_train)} samples...")
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = 0.0
        else:
            validation_data = None
        
        # Get callbacks
        callbacks = self.get_callbacks()
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose='auto' if verbose == 1 else 'silent'
        )
        
        logger.info("Training completed")
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X, verbose='silent')
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model on test data"""
        logger.info("Evaluating model...")
        
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        loss, mae, mse = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Calculate direction accuracy
        direction_accuracy = self._calculate_direction_accuracy(y_test, y_pred.flatten())
        
        metrics = {
            'loss': float(loss),
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'direction_accuracy': float(direction_accuracy)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    @staticmethod
    def _calculate_direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate directional accuracy (% of times model correctly predicts up/down movement)
        """
        if len(y_true) < 2:
            return 0.0
        
        # Calculate actual and predicted directions
        true_directions = np.diff(y_true) > 0
        pred_directions = np.diff(y_pred) > 0
        
        # Calculate accuracy
        correct = np.sum(true_directions == pred_directions)
        accuracy = correct / len(true_directions)
        
        return accuracy * 100  # Return as percentage
    
    def save(self, filepath: str):
        """Save model to file"""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file"""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def get_architecture_config(self) -> dict:
        """Get model architecture configuration"""
        return {
            'lstm_units_1': LSTM_UNITS_1,
            'lstm_units_2': LSTM_UNITS_2,
            'dense_units': DENSE_UNITS,
            'dropout_rate': DROPOUT_RATE,
            'learning_rate': self.learning_rate,
            'sequence_length': self.sequence_length
        }


def create_model(sequence_length: int, learning_rate: float = LEARNING_RATE) -> LSTMStockModel:
    """Factory function to create LSTM model"""
    return LSTMStockModel(sequence_length=sequence_length, learning_rate=learning_rate)


if __name__ == "__main__":
    # Test model creation
    sequence_length = 50
    model = create_model(sequence_length=sequence_length)
    
    # Print model configuration
    config = model.get_architecture_config()
    print("\nModel Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
