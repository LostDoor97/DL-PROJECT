"""
Utility script to run complete pipeline
Train multiple tickers, compare models, generate predictions
"""
import logging
import sys
from pathlib import Path

from config import TICKERS
from train import train_multiple_tickers
from baseline_arima import compare_models
from drift_monitor import monitor_all_tickers

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_pipeline(tickers=None, num_tickers=3):
    """
    Run the complete pipeline:
    1. Train LSTM models
    2. Train ARIMA baselines
    3. Compare models
    4. Generate drift reports
    """
    if tickers is None:
        tickers = TICKERS[:num_tickers]
    
    logger.info("="*70)
    logger.info("STARTING FULL ML PIPELINE")
    logger.info("="*70)
    logger.info(f"Processing tickers: {tickers}")
    
    results = {
        'lstm': {},
        'arima': {},
        'drift': {}
    }
    
    # Step 1: Train LSTM models
    logger.info("\n" + "#"*70)
    logger.info("# STEP 1: Training LSTM Models")
    logger.info("#"*70 + "\n")
    
    try:
        lstm_results = train_multiple_tickers(tickers, epochs=30)
        results['lstm'] = lstm_results
        logger.info(f"✅ Successfully trained LSTM for {len(lstm_results)} tickers")
    except Exception as e:
        logger.error(f"❌ LSTM training failed: {e}")
    
    # Step 2: Train ARIMA and compare (for first ticker)
    logger.info("\n" + "#"*70)
    logger.info("# STEP 2: ARIMA Baseline Comparison")
    logger.info("#"*70 + "\n")
    
    try:
        comparison_ticker = tickers[0]
        comparison_results = compare_models(comparison_ticker)
        results['comparison'] = comparison_results
        logger.info(f"✅ Model comparison completed for {comparison_ticker}")
    except Exception as e:
        logger.error(f"❌ Model comparison failed: {e}")
    
    # Step 3: Generate drift reports
    logger.info("\n" + "#"*70)
    logger.info("# STEP 3: Generating Drift Reports")
    logger.info("#"*70 + "\n")
    
    try:
        drift_results = monitor_all_tickers(tickers)
        results['drift'] = drift_results
        logger.info(f"✅ Drift reports generated for {len(drift_results)} tickers")
    except Exception as e:
        logger.error(f"❌ Drift monitoring failed: {e}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("="*70)
    
    logger.info(f"\n📊 LSTM Models Trained: {len(results['lstm'])}")
    for ticker, result in results['lstm'].items():
        logger.info(f"  {ticker}: Direction Accuracy = {result['metrics']['direction_accuracy']:.2f}%")
    
    if 'comparison' in results and results['comparison']:
        logger.info(f"\n🔬 Model Comparison:")
        for model_name, result in results['comparison'].items():
            logger.info(f"  {model_name}: {result['metrics']['direction_accuracy']:.2f}%")
    
    logger.info(f"\n📈 Drift Reports: {len(results['drift'])} generated")
    
    logger.info("\n" + "="*70)
    logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*70 + "\n")
    
    # Next steps
    logger.info("Next Steps:")
    logger.info("  1. View MLflow UI: mlflow ui --port 5000")
    logger.info("  2. Start API: python app.py")
    logger.info("  3. Launch Dashboard: streamlit run streamlit_app.py")
    logger.info("  4. Check drift reports in: evidently_reports/")
    
    return results


if __name__ == "__main__":
    # Parse command line arguments
    num_tickers = 3
    if len(sys.argv) > 1:
        try:
            num_tickers = int(sys.argv[1])
        except ValueError:
            logger.warning(f"Invalid argument: {sys.argv[1]}. Using default: 3 tickers")
    
    # Run pipeline
    results = run_full_pipeline(num_tickers=num_tickers)
