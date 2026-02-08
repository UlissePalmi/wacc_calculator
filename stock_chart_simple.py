"""
Stock Data to Excel
Fetches real stock data from Yahoo Finance and saves to Excel
All prices normalized to start at 100
"""

import yfinance as yf
import pandas as pd
import os

# ============================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================
TICKERS = {
    "ALLE": "ALLE",           # Allegion
    "SP500": "^GSPC",         # S&P 500 Index
    "Industrials": "XLI"      # Industrial Select Sector SPDR Fund
}
START_DATE = "2020-01-01"
END_DATE = "2026-02-04"
# ============================================

all_data = []

for name, ticker in TICKERS.items():
    print(f"Downloading {name} ({ticker}) data...")
    stock = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    
    if stock.empty:
        print(f"  No data found for {ticker}")
        continue
    
    print(f"  Downloaded {len(stock)} data points")
    
    # Flatten multi-index columns
    stock.columns = stock.columns.get_level_values(0)
    
    # Reset index and keep only Date and Close
    stock = stock.reset_index()[['Date', 'Close']]
    
    # Normalize to start at 100
    first_price = stock['Close'].iloc[0]
    stock['Close'] = (stock['Close'] / first_price) * 100
    
    stock.columns = ['Date', name]
    
    all_data.append(stock)

# Merge all dataframes on Date
result = all_data[0]
for df in all_data[1:]:
    result = pd.merge(result, df, on='Date', how='outer')

# Sort by date
result = result.sort_values('Date').reset_index(drop=True)

# Save to same directory as script
output_dir = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(output_dir, 'stock_data.xlsx')

result.to_excel(excel_path, index=False)
print(f"\nData saved to: {excel_path}")
print(f"Columns: {list(result.columns)}")
print("All prices normalized to start at 100")