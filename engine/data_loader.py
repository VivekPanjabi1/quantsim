import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


def load_data(symbol):
    """
    Load OHLCV data for a given symbol.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'MSFT')
    
    Returns:
        pd.DataFrame: DataFrame with Date as index and columns: Open, High, Low, Close, Volume
    """
    # Ensure data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    csv_path = os.path.join(data_dir, f"{symbol}.csv")
    
    # Check if CSV file exists
    if os.path.exists(csv_path):
        print(f"Loading existing data for {symbol} from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"Downloading data for {symbol} from Yahoo Finance")
        
        # Calculate date range (last 2 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2*365)
        
        # Download data using yfinance
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        
        # Select only the required columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")
    
    # Ensure Date column is datetime and set as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    elif df.index.name != 'Date':
        # If Date is already the index but not named properly
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'
    
    # Ensure we have the required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Sort by date to ensure chronological order
    df = df.sort_index()
    
    print(f"Loaded {len(df)} rows of data for {symbol}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df
