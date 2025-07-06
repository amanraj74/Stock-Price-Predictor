import argparse
from src.train import run_training

def main():
    parser = argparse.ArgumentParser(description='Stock Price Predictor')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker (e.g., AAPL)')
    parser.add_argument('--start_date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='End date (YYYY-MM-DD)')
    args = parser.parse_args()

    run_training(args.ticker.upper(), args.start_date, args.end_date)

if __name__ == "__main__":
    main() 