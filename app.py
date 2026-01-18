"""
FastAPI Application for Stock Price Prediction
Endpoint: /predict?ticker=MCC
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import logging
from pathlib import Path
import uvicorn

from predict import StockPredictor, find_latest_model
from config import API_HOST, API_PORT, PREDICTION_STEPS, TICKERS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Stock Price Prediction API",
    description="LSTM-based stock price prediction with EMA smoothing",
    version="1.0.0"
)


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    ticker: str
    prediction_steps: int
    raw_predictions: List[float]
    smoothed_predictions: List[float]
    last_actual_value: float
    latest_date: str
    model_path: str
    status: str = "success"


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return {
        "status": "healthy",
        "message": "Stock Price Prediction API is running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "API is operational"
    }


@app.get("/tickers")
async def list_tickers():
    """List available tickers"""
    return {
        "tickers": TICKERS,
        "count": len(TICKERS)
    }


@app.get("/predict", response_model=PredictionResponse)
async def predict(
    ticker: str = Query(..., description="Stock ticker symbol (e.g., MCC.V, AAPL)"),
    steps: int = Query(PREDICTION_STEPS, ge=1, le=100, description="Number of steps to predict ahead"),
    smoothing: bool = Query(True, description="Apply EMA smoothing to predictions")
):
    """
    Predict stock prices for a given ticker
    
    Args:
        ticker: Stock ticker symbol
        steps: Number of steps to predict ahead (default: 30)
        smoothing: Whether to apply EMA smoothing (default: True)
        
    Returns:
        Prediction results with raw and smoothed predictions
    """
    try:
        logger.info(f"Received prediction request for ticker: {ticker}, steps: {steps}")
        
        # Find model for ticker
        model_path = find_latest_model(ticker)
        
        if model_path is None:
            raise HTTPException(
                status_code=404,
                detail=f"No trained model found for ticker {ticker}. Please train a model first."
            )
        
        # Create predictor
        predictor = StockPredictor(model_path, ticker)
        
        # Generate predictions
        result = predictor.predict_from_ticker(steps=steps, apply_smoothing=smoothing)
        
        logger.info(f"Prediction successful for {ticker}")
        
        return PredictionResponse(**result, status="success")
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        logger.error(f"Prediction error for {ticker}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating predictions: {str(e)}"
        )


@app.get("/models/{ticker}")
async def get_model_info(ticker: str):
    """Get information about the model for a specific ticker"""
    try:
        model_path = find_latest_model(ticker)
        
        if model_path is None:
            raise HTTPException(
                status_code=404,
                detail=f"No model found for ticker {ticker}"
            )
        
        model_file = Path(model_path)
        
        return {
            "ticker": ticker,
            "model_path": str(model_path),
            "model_name": model_file.name,
            "model_size_mb": model_file.stat().st_size / (1024 * 1024),
            "last_modified": model_file.stat().st_mtime
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(
    tickers: List[str],
    steps: int = Query(PREDICTION_STEPS, ge=1, le=100),
    smoothing: bool = Query(True)
):
    """
    Batch prediction for multiple tickers
    
    Args:
        tickers: List of stock ticker symbols
        steps: Number of steps to predict ahead
        smoothing: Whether to apply EMA smoothing
        
    Returns:
        Dictionary of predictions for each ticker
    """
    results = {}
    errors = {}
    
    for ticker in tickers:
        try:
            model_path = find_latest_model(ticker)
            
            if model_path is None:
                errors[ticker] = "No trained model found"
                continue
            
            predictor = StockPredictor(model_path, ticker)
            result = predictor.predict_from_ticker(steps=steps, apply_smoothing=smoothing)
            results[ticker] = result
            
        except Exception as e:
            logger.error(f"Error predicting {ticker}: {e}")
            errors[ticker] = str(e)
    
    return {
        "predictions": results,
        "errors": errors,
        "success_count": len(results),
        "error_count": len(errors)
    }


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "status": "error",
            "message": "Resource not found",
            "detail": str(exc.detail) if hasattr(exc, 'detail') else "Not found"
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler"""
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "detail": str(exc.detail) if hasattr(exc, 'detail') else "An error occurred"
        }
    )


if __name__ == "__main__":
    logger.info(f"Starting FastAPI server on {API_HOST}:{API_PORT}")
    uvicorn.run(
        "app:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )
