import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime

# Optional: For LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Create data directory if not exists
data_dir = 'data'
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
    df['Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    df = df.dropna()
    return df

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(12,6))
    plt.plot(y_true.index, y_true, label='Actual')
    plt.plot(y_true.index, y_pred, label='Predicted')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

def prepare_lstm_data(series, look_back=10):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i+look_back])
        y.append(series[i+look_back])
    return np.array(X), np.array(y)

def train_lstm(X_train, y_train, X_val, y_val):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), callbacks=[es], verbose=0)
    return model

def main():
    print("\n=== Stock Price Predictor ===\n")
    ticker = input("Enter stock ticker (e.g., AAPL): ").upper()
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    # Validate date input
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")
        return

    # Download and preprocess data
    try:
        df = download_stock_data(ticker, start_date, end_date)
        if df.empty:
            print("No data found for the given ticker and date range.")
            return
    except Exception as e:
        print(f"Error downloading data: {e}")
        return

    df_feat = create_features(df)
    features = ['Return', 'MA5', 'MA10', 'MA20', 'Volume_Change']
    X = df_feat[features]
    y = df_feat['Close']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # Linear Regression
    print("\nTraining Linear Regression model...")
    lr_model = train_linear_regression(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    print(f"Linear Regression RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}")
    plot_predictions(y_test, y_pred_lr, f"{ticker} - Linear Regression")
    # Save predictions to CSV
    pd.DataFrame({"Actual": y_test, "Predicted": y_pred_lr}, index=y_test.index).to_csv(f"{ticker}_linear_regression_predictions.csv")

    # LSTM Model
    print("\nTraining LSTM model...")
    scaler = MinMaxScaler()
    close_scaled = scaler.fit_transform(df_feat['Close'].values.reshape(-1,1))
    look_back = 10
    X_lstm, y_lstm = prepare_lstm_data(close_scaled, look_back)
    split = int(len(X_lstm)*0.8)
    X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
    y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]
    X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
    X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1))
    lstm_model = train_lstm(X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm)
    y_pred_lstm = lstm_model.predict(X_test_lstm)
    y_pred_lstm_rescaled = scaler.inverse_transform(y_pred_lstm)
    y_test_lstm_rescaled = scaler.inverse_transform(y_test_lstm.reshape(-1,1))
    print(f"LSTM RMSE: {np.sqrt(mean_squared_error(y_test_lstm_rescaled, y_pred_lstm_rescaled)):.2f}")
    # For plotting, align indices
    plot_idx = df_feat.index[-len(y_test_lstm_rescaled):]
    plot_predictions(pd.Series(y_test_lstm_rescaled.flatten(), index=plot_idx), y_pred_lstm_rescaled.flatten(), f"{ticker} - LSTM")
    # Save LSTM predictions to CSV
    pd.DataFrame({"Actual": y_test_lstm_rescaled.flatten(), "Predicted": y_pred_lstm_rescaled.flatten()}, index=plot_idx).to_csv(f"{ticker}_lstm_predictions.csv")

if __name__ == "__main__":
    main() 