import yfinance as yf
import pandas as pd
import time

# Function to fetch stock data
def fetch_stock_data(symbol, start_date="2024-06-01", end_date="2024-08-01"):
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    return data

# Save stock data to CSV
def save_stock_data(symbol, data):
    filename = f"{symbol}_last_two_months_data.csv"
    data.to_csv(filename)
    print(f"Data saved to {filename}")

# List of symbols to fetch
symbols = ["AAPL", "005930.KS", "X", "MSFT", "GOOGL", "TSLA","X", "AMZN" , "META" , "NVDA", "JNJ" , "7203.T" , "6758.T" , "NESN.SW","TCS" ,"INFY"]  # Add more symbols as needed

# Fetch and save stock data for each symbol
for symbol in symbols:
    try:
        print(f"Fetching data for {symbol}...")
        data = fetch_stock_data(symbol)
        save_stock_data(symbol, data)
        print(f"Data for {symbol} fetched and saved successfully.")
        time.sleep(1)  # Shorter sleep time as yfinance generally has higher limits
    except Exception as e:
        print(e)
        print(f"Waiting 60 seconds before retrying...")
        time.sleep(60)  # Wait longer before retrying in case of an error
