"""
Drift Monitoring Module using Evidently
Monitors data drift and model drift
"""
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from typing import Optional

# Simplified version without Evidently for compatibility
EVIDENTLY_AVAILABLE = False
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, RegressionPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Evidently not available. Using simplified drift monitoring.")

from data_pipeline import StockDataPipeline
from predict import StockPredictor, find_latest_model
from config import REPORTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftMonitor:
    """Monitor data and model drift using Evidently"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.reports_dir = REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def create_drift_report(self, reference_data: pd.DataFrame, 
                           current_data: pd.DataFrame,
                           report_name: Optional[str] = None) -> str:
        """
        Create data drift report comparing reference and current data
        
        Args:
            reference_data: Reference dataset (e.g., training data)
            current_data: Current dataset (e.g., recent production data)
            report_name: Name for the report file
            
        Returns:
            Path to the generated HTML report
        """
        logger.info(f"Generating data drift report for {self.ticker}...")
        
        if report_name is None:
            report_name = f"{self.ticker}_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        report_path = self.reports_dir / report_name
        
        if EVIDENTLY_AVAILABLE:
            try:
                # Create drift report
                report = Report(metrics=[DataDriftPreset()])
                
                # Run report
                report.run(reference_data=reference_data, current_data=current_data)
                
                # Save report
                report.save_html(str(report_path))
            except Exception as e:
                logger.warning(f"Evidently report failed: {e}. Creating simplified report.")
                self._create_simple_drift_report(reference_data, current_data, report_path)
        else:
            self._create_simple_drift_report(reference_data, current_data, report_path)
        
        logger.info(f"Drift report saved to {report_path}")
        return str(report_path)
    
    def _create_simple_drift_report(self, reference_data: pd.DataFrame, 
                                   current_data: pd.DataFrame, 
                                   report_path: Path):
        """Create simple HTML drift report without Evidently"""
        # Calculate basic statistics
        ref_mean = reference_data.mean().values[0]
        ref_std = reference_data.std().values[0]
        curr_mean = current_data.mean().values[0]
        curr_std = current_data.std().values[0]
        
        mean_drift = abs(curr_mean - ref_mean) / ref_std if ref_std > 0 else 0
        
        html = f"""
        <html>
        <head><title>Drift Report - {self.ticker}</title></head>
        <body>
        <h1>Data Drift Report: {self.ticker}</h1>
        <h2>Summary Statistics</h2>
        <table border="1">
        <tr><th>Metric</th><th>Reference</th><th>Current</th><th>Drift</th></tr>
        <tr><td>Mean</td><td>{ref_mean:.6f}</td><td>{curr_mean:.6f}</td><td>{mean_drift:.4f}</td></tr>
        <tr><td>Std Dev</td><td>{ref_std:.6f}</td><td>{curr_std:.6f}</td><td>-</td></tr>
        <tr><td>Samples</td><td>{len(reference_data)}</td><td>{len(current_data)}</td><td>-</td></tr>
        </table>
        <p>Generated: {datetime.now()}</p>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html)
    
    def create_model_performance_report(self, reference_data: pd.DataFrame,
                                       current_data: pd.DataFrame,
                                       reference_predictions: np.ndarray,
                                       current_predictions: np.ndarray,
                                       report_name: Optional[str] = None) -> str:
        """
        Create model performance report
        
        Args:
            reference_data: Reference dataset with actual values
            current_data: Current dataset with actual values
            reference_predictions: Predictions on reference data
            current_predictions: Predictions on current data
            report_name: Name for the report file
            
        Returns:
            Path to the generated HTML report
        """
        logger.info(f"Generating model performance report for {self.ticker}...")
        
        if report_name is None:
            report_name = f"{self.ticker}_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        report_path = self.reports_dir / report_name
        
        if EVIDENTLY_AVAILABLE:
            try:
                # Prepare data with predictions
                ref_df = reference_data.copy()
                ref_df['prediction'] = reference_predictions
                
                curr_df = current_data.copy()
                curr_df['prediction'] = current_predictions
                
                # Create report
                report = Report(metrics=[RegressionPreset()])
                
                # Run report
                report.run(reference_data=ref_df, current_data=curr_df)
                
                # Save report
                report.save_html(str(report_path))
            except Exception as e:
                logger.warning(f"Evidently report failed: {e}. Creating simplified report.")
                self._create_simple_performance_report(reference_data, current_data, 
                                                      reference_predictions, current_predictions, 
                                                      report_path)
        else:
            self._create_simple_performance_report(reference_data, current_data, 
                                                  reference_predictions, current_predictions, 
                                                  report_path)
        
        logger.info(f"Performance report saved to {report_path}")
        return str(report_path)
    
    def _create_simple_performance_report(self, reference_data: pd.DataFrame,
                                        current_data: pd.DataFrame,
                                        reference_predictions: np.ndarray,
                                        current_predictions: np.ndarray,
                                        report_path: Path):
        """Create simple HTML performance report without Evidently"""
        ref_actual = reference_data.values.flatten()
        curr_actual = current_data.values.flatten()
        
        ref_mse = np.mean((ref_actual - reference_predictions) ** 2)
        curr_mse = np.mean((curr_actual - current_predictions) ** 2)
        
        html = f"""
        <html>
        <head><title>Performance Report - {self.ticker}</title></head>
        <body>
        <h1>Model Performance Report: {self.ticker}</h1>
        <h2>Prediction Metrics</h2>
        <table border="1">
        <tr><th>Metric</th><th>Reference</th><th>Current</th></tr>
        <tr><td>MSE</td><td>{ref_mse:.6f}</td><td>{curr_mse:.6f}</td></tr>
        <tr><td>RMSE</td><td>{np.sqrt(ref_mse):.6f}</td><td>{np.sqrt(curr_mse):.6f}</td></tr>
        </table>
        <p>Generated: {datetime.now()}</p>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html)
    
    def monitor_ticker(self, lookback_days: int = 30) -> dict:
        """
        Complete drift monitoring for a ticker
        
        Args:
            lookback_days: Number of days to look back for current data
            
        Returns:
            Dictionary with report paths
        """
        logger.info(f"Starting drift monitoring for {self.ticker}...")
        
        try:
            # Fetch data
            pipeline = StockDataPipeline(self.ticker)
            pipeline.fetch_data()
            data = pipeline.compute_mid_price()
            
            # Split into reference (older) and current (recent)
            split_idx = -lookback_days if len(data) > lookback_days else len(data) // 2
            
            reference_data = data.iloc[:split_idx][['Mid']].copy()
            current_data = data.iloc[split_idx:][['Mid']].copy()
            
            reference_data.columns = ['target']
            current_data.columns = ['target']
            
            # Create drift report
            drift_report_path = self.create_drift_report(reference_data, current_data)
            
            # Get predictions if model exists
            model_path = find_latest_model(self.ticker)
            performance_report_path = None
            
            if model_path:
                try:
                    predictor = StockPredictor(model_path, self.ticker)
                    
                    # Prepare sequences for prediction
                    normalized_prices = pipeline.windowed_normalize(np.array(data['Mid'].values))
                    from config import UNROLLINGS
                    
                    # Get predictions for reference and current periods
                    ref_normalized = normalized_prices[:split_idx]
                    curr_normalized = normalized_prices[split_idx:]
                    
                    if len(ref_normalized) > UNROLLINGS and len(curr_normalized) > UNROLLINGS:
                        # Simple single-step predictions for comparison
                        ref_input = predictor.prepare_input_sequence(ref_normalized)
                        curr_input = predictor.prepare_input_sequence(curr_normalized)
                        
                        ref_pred = predictor.predict_single_step(ref_input)
                        curr_pred = predictor.predict_single_step(curr_input)
                        
                        # Create simple comparison
                        ref_preds = np.array([ref_pred])
                        curr_preds = np.array([curr_pred])
                        
                        # Only create performance report if we have enough data
                        if len(reference_data) > 0 and len(current_data) > 0:
                            logger.info("Creating performance report...")
                            # Use subset of data for report
                            ref_subset = reference_data.tail(1)
                            curr_subset = current_data.tail(1)
                            
                            performance_report_path = self.create_model_performance_report(
                                ref_subset, curr_subset,
                                ref_preds[:len(ref_subset)],
                                curr_preds[:len(curr_subset)]
                            )
                
                except Exception as e:
                    logger.warning(f"Could not generate performance report: {e}")
            
            results = {
                'ticker': self.ticker,
                'drift_report': drift_report_path,
                'performance_report': performance_report_path,
                'reference_samples': len(reference_data),
                'current_samples': len(current_data)
            }
            
            logger.info(f"Drift monitoring completed for {self.ticker}")
            return results
            
        except Exception as e:
            logger.error(f"Error monitoring {self.ticker}: {e}")
            raise


def monitor_all_tickers(tickers: Optional[list] = None) -> dict:
    """Monitor drift for all tickers"""
    from config import TICKERS
    
    if tickers is None:
        tickers = TICKERS[:3]  # Monitor first 3 tickers
    
    results = {}
    
    for ticker in tickers:
        try:
            monitor = DriftMonitor(ticker)
            result = monitor.monitor_ticker()
            results[ticker] = result
        except Exception as e:
            logger.error(f"Failed to monitor {ticker}: {e}")
            continue
    
    return results


if __name__ == "__main__":
    # Monitor a single ticker
    ticker = "MCC.V"
    monitor = DriftMonitor(ticker)
    result = monitor.monitor_ticker(lookback_days=30)
    
    print(f"\nDrift Monitoring Results for {ticker}:")
    print(f"Drift Report: {result['drift_report']}")
    if result['performance_report']:
        print(f"Performance Report: {result['performance_report']}")
