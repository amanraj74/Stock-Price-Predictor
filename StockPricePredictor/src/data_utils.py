import os
import pandas as pd
import yfinance as yf

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(data_dir, exist_ok=True)

def download_stock_data(ticker, start_date, end_date):
    """Download historical stock data from Yahoo Finance."""
    file_path = os.path.join(data_dir, f"{ticker}_{start_date}_{end_date}.csv")
    if os.path.exists(file_path):
        print(f"Loading cached data from {file_path}")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        print(f"Downloading data for {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date)
        df.to_csv(file_path)
    return df

def create_features(df):
    """Create features for the model."""
    df = df.copy()
    # Ensure numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df = df.dropna(subset=['Close', 'Volume'])
    df['Return'] = df['Close'].pct_change(fill_method=None)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Volume_Change'] = df['Volume'].pct_change(fill_method=None)
    df = df.dropna()
    return df 