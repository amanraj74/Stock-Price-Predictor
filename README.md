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

![Sample Plot](stockpricepredictor/sample_plot.png)

## Project Structure
- `main.py`: CLI entry point for training and prediction
- `src/`: Source code modules
  - `data_utils.py`: Data download and feature engineering
  - `models.py`: Model definitions
  - `train.py`: Training and evaluation logic
- `notebooks/`: Jupyter notebooks for EDA
- `data/`: Downloaded stock data (in .gitignore)
- `models/`: Saved models (in .gitignore)
- `requirements.txt`: Dependencies
- `LICENSE`: License file
- `.gitignore`: Ignore data, models, etc.

## Setup
1. Clone the repository or copy the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main script from the project root:
```bash
python main.py --ticker AAPL --start_date 2020-01-01 --end_date 2023-01-01
```

## Author
Aman Jaiswal


[samplePlotImage]: stockpricepredictor/sample_plot.png
