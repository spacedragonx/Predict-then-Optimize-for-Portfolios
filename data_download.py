import time
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

output_folder = "data_parquet"
os.makedirs(output_folder, exist_ok=True)


# --- CONFIGURATION ---
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
# Use 'SIP' if you want full market data (delayed 15m, better for training)
# Use 'IEX' if you want real-time but with missing data (worse for training)
FEED = 'sip' 
SYMBOLS = ["AAPL", "MSFT", "TSLA"] 
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2023, 1, 1)

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

def download_and_save(symbol):
    print(f"Downloading {symbol}...")
    
    # Alpaca allows retrieving huge chunks if we don't slice it too small
    # We ask for the full range; the SDK handles pagination automatically
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,  # 1-Minute resolution
        start=START_DATE,
        end=END_DATE,
        feed=FEED 
    )
    
    try:
        # The SDK automatically makes multiple requests if data > 10,000 bars
        # This handles the pagination logic for you.
        bars = client.get_stock_bars(request_params)
        
        # Convert to DataFrame
        df = bars.df
        
        # Save to CSV
        filename = f"{output_folder}/{symbol}_1min.parquet"
        df.to_parquet(filename, compression="snappy")
        
        print(f"Saved {len(df)} rows to {filename}")
        
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")

# --- EXECUTION LOOP ---
for symbol in SYMBOLS:
    download_and_save(symbol)  
    # Even though we are likely safe, sleep 0.5s between symbols 
    time.sleep(0.5)