"""
Generate synthetic stock data for testing when yfinance is unavailable
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_stock_data(ticker: str, start_date: str = '2020-01-01', 
                                  end_date: Optional[str] = None, num_days: int = 1500) -> pd.DataFrame:
    """
    Generate realistic synthetic stock price data
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date
        end_date: End date (optional)
        num_days: Number of trading days to generate
        
    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Generating synthetic data for {ticker}...")
    
    # Parse dates
    start = pd.to_datetime(start_date)
    if end_date:
        end = pd.to_datetime(end_date)
        num_days = (end - start).days
    else:
        end = start + timedelta(days=num_days)
    
    # Generate date range (trading days only)
    dates = pd.bdate_range(start=start, end=end)[:num_days]
    
    # Initialize price with random walk
    np.random.seed(hash(ticker) % 2**32)  # Deterministic based on ticker
    
    # Starting price based on ticker
    base_prices = {
        'AAPL': 150.0,
        'GOOGL': 2800.0,
        'MSFT': 300.0,
        'TSLA': 800.0,
        'NVDA': 500.0,
        'AMZN': 3300.0,
        'META': 350.0,
    }
    
    initial_price = base_prices.get(ticker, 100.0)
    
    # Generate realistic price movements
    daily_returns = np.random.normal(0.0005, 0.02, len(dates))  # Mean return, volatility
    
    # Add trend
    trend = np.linspace(0, 0.3, len(dates))  # Slight upward trend
    daily_returns += trend / len(dates)
    
    # Calculate prices
    price_multipliers = np.cumprod(1 + daily_returns)
    close_prices = initial_price * price_multipliers
    
    # Generate OHLC from close
    daily_volatility = np.random.uniform(0.005, 0.03, len(dates))
    
    high_prices = close_prices * (1 + daily_volatility)
    low_prices = close_prices * (1 - daily_volatility)
    
    # Open is previous close (with small gap)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = initial_price
    open_prices = open_prices * (1 + np.random.normal(0, 0.005, len(dates)))
    
    # Volume
    base_volume = 50_000_000  # 50M average volume
    volume = np.random.lognormal(np.log(base_volume), 0.5, len(dates)).astype(int)
    
    # Adjusted Close (same as close for simplicity)
    adj_close = close_prices.copy()
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Adj Close': adj_close,
        'Volume': volume
    })
    
    logger.info(f"Generated {len(df)} days of synthetic data for {ticker}")
    logger.info(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    
    return df


def save_synthetic_data(ticker: str, output_dir: str = 'data/raw'):
    """Generate and save synthetic data"""
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = generate_synthetic_stock_data(ticker)
    
    output_file = output_path / f"{ticker}_raw.csv"
    df.to_csv(output_file, index=False)
    
    logger.info(f"Saved synthetic data to {output_file}")
    return output_file


if __name__ == "__main__":
    # Generate data for all tickers
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN']
    
    for ticker in tickers:
        save_synthetic_data(ticker)
    
    print("\nSynthetic data generated for all tickers!")
