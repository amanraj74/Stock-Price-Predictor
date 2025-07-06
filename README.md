# Stock Price Predictor

Predict stock prices using historical data and machine learning.

## Features
- Download historical stock data from Yahoo Finance
- Data preprocessing and feature engineering
- Linear Regression and LSTM models for prediction
- Visualization of actual vs. predicted prices

## Sample Results
- **Linear Regression RMSE:** ~5.23 (example)
- **LSTM RMSE:** ~3.87 (example)

![Sample Plot](sample_plot.png)

## Setup
1. Clone the repository or copy the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the main script:
   ```bash
   python main.py
   ```
2. Follow the prompts to enter a stock ticker and date range.

## Project Structure
- `main.py`: Main script to run the project
- `data/`: Folder for storing downloaded stock data
- `notebooks/`: Jupyter notebooks for exploration
- `models/`: Saved models

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- yfinance
- tensorflow (for LSTM)

## Author
Aman Jaiswal
