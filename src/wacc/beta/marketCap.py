import yfinance as yf
import pandas as pd

# --- 1. Your Manual Data ---
ALLE_CAPITAL_STRUCTURE = {
    # 2019
    '2019-Q1': {'total_debt': 1450, 'tax_rate': 0.16},
    '2019-Q2': {'total_debt': 1445, 'tax_rate': 0.16},
    '2019-Q3': {'total_debt': 1440, 'tax_rate': 0.16},
    '2019-Q4': {'total_debt': 1435, 'tax_rate': 0.154},
    # 2020
    '2020-Q1': {'total_debt': 1430, 'tax_rate': 0.15},
    '2020-Q2': {'total_debt': 1425, 'tax_rate': 0.15},
    '2020-Q3': {'total_debt': 1420, 'tax_rate': 0.15},
    '2020-Q4': {'total_debt': 1415, 'tax_rate': 0.148},
    # 2021
    '2021-Q1': {'total_debt': 1410, 'tax_rate': 0.15},
    '2021-Q2': {'total_debt': 1400, 'tax_rate': 0.15},
    '2021-Q3': {'total_debt': 1395, 'tax_rate': 0.15},
    '2021-Q4': {'total_debt': 1390, 'tax_rate': 0.124},
    # 2022
    '2022-Q1': {'total_debt': 1385, 'tax_rate': 0.13},
    '2022-Q2': {'total_debt': 1380, 'tax_rate': 0.13},
    '2022-Q3': {'total_debt': 2100, 'tax_rate': 0.13},
    '2022-Q4': {'total_debt': 2050, 'tax_rate': 0.122},
    # 2023
    '2023-Q1': {'total_debt': 2020, 'tax_rate': 0.14},
    '2023-Q2': {'total_debt': 1995, 'tax_rate': 0.14},
    '2023-Q3': {'total_debt': 1980, 'tax_rate': 0.14},
    '2023-Q4': {'total_debt': 1965, 'tax_rate': 0.138},
    # 2024
    '2024-Q1': {'total_debt': 1990, 'tax_rate': 0.145},
    '2024-Q2': {'total_debt': 2000, 'tax_rate': 0.145},
    '2024-Q3': {'total_debt': 1995, 'tax_rate': 0.145},
    '2024-Q4': {'total_debt': 1999.5, 'tax_rate': 0.145},
    # 2025
    '2025-Q1': {'total_debt': 2100, 'tax_rate': 0.145},
}

# --- 2. Helper: Get Market Cap (From Yahoo) ---
def get_market_cap(symbol, start_date):
    try:
        ticker = yf.Ticker(symbol)
        shares = ticker.get_shares_full(start=start_date)
        prices = ticker.history(start=start_date)
        
        if shares.empty or prices.empty: return pd.DataFrame()
        
        # Align Timezones
        shares.index = pd.to_datetime(shares.index).normalize().tz_localize(None)
        prices.index = pd.to_datetime(prices.index).normalize().tz_localize(None)
        
        # Merge Shares and Price
        shares = shares.groupby(shares.index).last()
        aligned_shares = shares.reindex(prices.index, method='ffill')
        
        # Calculate Market Cap in Millions
        market_cap = (aligned_shares * prices['Close']) / 1_000_000
        
        df = pd.DataFrame({'Market Cap': market_cap})
        df.index.name = 'Date'
        return df
    except Exception as e:
        print(f"MC Error: {e}")
        return pd.DataFrame()

# --- 3. Helper: Process Manual Data ---
def process_manual_data(manual_dict):
    """Converts the '2019-Q1' dictionary into a proper DateTime DataFrame."""
    data_list = []
    
    # Map Quarters to approximate end dates
    q_map = {
        'Q1': '-03-31', 
        'Q2': '-06-30', 
        'Q3': '-09-30', 
        'Q4': '-12-31'
    }

    for quarter_str, values in manual_dict.items():
        year, q = quarter_str.split('-')
        date_str = year + q_map[q]
        
        data_list.append({
            'Date': pd.Timestamp(date_str),
            'Total Debt': values['total_debt'],
            'Tax Rate': values['tax_rate']
        })
    
    # Create DataFrame
    df = pd.DataFrame(data_list)
    df = df.set_index('Date').sort_index()
    return df

# --- 4. Main Function: Merge Everything ---
def get_wacc_inputs(symbol, start_date, manual_data):
    # A. Get Daily Market Cap (The "Skeleton")
    df_mc = get_market_cap(symbol, start_date)
    if df_mc.empty: 
        print("Error: No Market Cap data found.")
        return pd.DataFrame()

    # B. Get Manual Debt & Tax
    df_manual = process_manual_data(manual_data)
    
    # C. Merge
    # We join the manual quarterly data onto the daily market cap data
    combined = df_mc.join(df_manual, how='left')
    
    # D. Fill Gaps
    # Forward fill: applies the Q1 numbers to every day until Q2 starts
    combined['Total Debt'] = combined['Total Debt'].ffill()
    combined['Tax Rate'] = combined['Tax Rate'].ffill()
    
    # Backfill: Ensures the days in Jan/Feb before the first March report have data
    combined['Total Debt'] = combined['Total Debt'].bfill()
    combined['Tax Rate'] = combined['Tax Rate'].bfill()
    combined['Market Cap'] = combined['Market Cap'].bfill()

    return combined