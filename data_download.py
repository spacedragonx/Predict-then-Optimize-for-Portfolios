import time
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

output_folder = "data_csv"
os.makedirs(output_folder, exist_ok=True)


# --- CONFIGURATION ---
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

FEED = 'sip' 
SYMBOLS = ['AAPL', 'MSFT', 'AMZN', 'JPM', 'XOM',
    'JNJ', 'KO', 'WMT', 'PG', 'IBM',
    'INTC', 'CSCO', 'ORCL', 'HD', 'MCD'] 
START_DATE = datetime(2000, 1, 1)
END_DATE = datetime(2026, 1, 1)

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

def download_and_save(symbol):
    print(f"Downloading data...")
    
    all_symbol = symbol + ["SPY"]
    request_params = StockBarsRequest(
        symbol_or_symbols=all_symbol,
        timeframe=TimeFrame.Day,  
        start=START_DATE,
        end=END_DATE,
        feed=FEED 
    )
    
    try:
        
        bars = client.get_stock_bars(request_params)
        
        # Convert to DataFrame
        df = bars.df
        
        # Save to CSV
        filename = f"{output_folder}/25yr_daily_data_with_spy.csv"
        df.to_csv(filename)
        
        print(f"Saved {len(df)} rows to {filename}")
        
    except Exception as e:
        print(f"Error downloading data: {e}")

# --- EXECUTION ---

download_and_save(SYMBOLS)  

