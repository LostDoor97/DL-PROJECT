"""
Quick test script to verify yfinance data fetching
"""
import yfinance as yf
import pandas as pd
from datetime import datetime

print("Testing yfinance data fetching...")
print(f"yfinance version: {yf.__version__}")
print()

tickers_to_test = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

for ticker in tickers_to_test:
    print(f"Testing {ticker}...")
    try:
        # Method 1: Date range
        data1 = yf.download(ticker, start='2024-01-01', end='2024-12-31', progress=False)
        print(f"  Method 1 (date range): {len(data1)} rows")
        
        # Method 2: Period
        data2 = yf.download(ticker, period='1y', progress=False)
        print(f"  Method 2 (period='1y'): {len(data2)} rows")
        
        # Show columns
        if not data2.empty:
            print(f"  Columns: {data2.columns.tolist()}")
            print(f"  Sample data:")
            print(data2.head(2))
        
        print(f"  ✓ {ticker} OK")
        
    except Exception as e:
        print(f"  ✗ {ticker} FAILED: {e}")
    
    print()

print("Test complete!")
