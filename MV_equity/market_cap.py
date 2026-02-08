from wacc.beta import marketCap
import yfinance as yf
import pandas as pd

# --- Usage ---
ticker = input("Enter ticker symbol: ").upper()
start_date = "2019-01-01"

wacc_df = marketCap.get_wacc_inputs(ticker, start_date, marketCap.ALLE_CAPITAL_STRUCTURE)

print(f"--- Final WACC Inputs for {ticker} ---")
pd.options.display.float_format = '{:,.2f}'.format
print(wacc_df.head(10))  # Should show 2019 data filled
print(wacc_df.tail(10))  # Should show 2024/2025 data filled

import os

# 1. Define folder and filename
output_folder = "data"
output_file = "marketCap.csv"

# 2. Create the folder if it doesn't exist (Safety Check)
os.makedirs(output_folder, exist_ok=True)

# 3. Save the CSV inside that folder
wacc_df.to_csv(f"{output_folder}/{output_file}")

print(f"Saved successfully to {output_folder}/{output_file}")