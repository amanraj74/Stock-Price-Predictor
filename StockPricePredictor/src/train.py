import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from src.data_utils import download_stock_data, create_features
from src.models import train_linear_regression, build_lstm, train_lstm
import matplotlib.pyplot as plt

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

def run_training(ticker, start_date, end_date):
    # Download and preprocess data
    df = download_stock_data(ticker, start_date, end_date)
    if df.empty:
        print("No data found for the given ticker and date range.")
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
    pd.DataFrame({"Actual": y_test, "Predicted": np.ravel(y_pred_lr)}, index=y_test.index).to_csv(f"{ticker}_linear_regression_predictions.csv")

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
    lstm_model = build_lstm((look_back, 1))
    lstm_model = train_lstm(lstm_model, X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm)
    y_pred_lstm = lstm_model.predict(X_test_lstm)
    y_pred_lstm_rescaled = scaler.inverse_transform(y_pred_lstm)
    y_test_lstm_rescaled = scaler.inverse_transform(y_test_lstm.reshape(-1,1))
    print(f"LSTM RMSE: {np.sqrt(mean_squared_error(y_test_lstm_rescaled, y_pred_lstm_rescaled)):.2f}")
    plot_idx = df_feat.index[-len(y_test_lstm_rescaled):]
    plot_predictions(pd.Series(y_test_lstm_rescaled.flatten(), index=plot_idx), y_pred_lstm_rescaled.flatten(), f"{ticker} - LSTM")
    pd.DataFrame({"Actual": y_test_lstm_rescaled.flatten(), "Predicted": y_pred_lstm_rescaled.flatten()}, index=plot_idx).to_csv(f"{ticker}_lstm_predictions.csv") 