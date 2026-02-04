"""
Pull ALLE historical market cap (market value of equity) for each quarter end.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Quarter end dates
QUARTER_ENDS = {
    '2019-Q1': '2019-03-31',
    '2019-Q2': '2019-06-30',
    '2019-Q3': '2019-09-30',
    '2019-Q4': '2019-12-31',
    '2020-Q1': '2020-03-31',
    '2020-Q2': '2020-06-30',
    '2020-Q3': '2020-09-30',
    '2020-Q4': '2020-12-31',
    '2021-Q1': '2021-03-31',
    '2021-Q2': '2021-06-30',
    '2021-Q3': '2021-09-30',
    '2021-Q4': '2021-12-31',
    '2022-Q1': '2022-03-31',
    '2022-Q2': '2022-06-30',
    '2022-Q3': '2022-09-30',
    '2022-Q4': '2022-12-31',
    '2023-Q1': '2023-03-31',
    '2023-Q2': '2023-06-30',
    '2023-Q3': '2023-09-30',
    '2023-Q4': '2023-12-31',
    '2024-Q1': '2024-03-31',
    '2024-Q2': '2024-06-30',
    '2024-Q3': '2024-09-30',
    '2024-Q4': '2024-12-31',
    '2025-Q1': '2025-01-31',  # Use Jan 31 as proxy for Q1 2025 (quarter not over)
}

# Approximate shares outstanding (millions) - from 10-K/10-Q filings
# These are relatively stable for ALLE
SHARES_OUTSTANDING = {
    '2019': 93.0,
    '2020': 92.0,
    '2021': 90.5,
    '2022': 89.0,
    '2023': 88.0,
    '2024': 87.0,
    '2025': 86.5,
}

def get_quarter_end_price(ticker: str, date_str: str) -> float:
    """Get closing price on or near quarter end date."""
    
    target_date = pd.Timestamp(date_str)
    
    # Download a small window around the target date
    start = target_date - timedelta(days=10)
    end = target_date + timedelta(days=5)
    
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        
        if data.empty:
            return None
        
        # Get the Close column (handle multi-index)
        close = data['Close'].squeeze() if 'Close' in data.columns else data['Close']
        
        # Find closest date to target
        close.index = pd.to_datetime(close.index)
        
        # Get price on or before target date
        valid_dates = close.index[close.index <= target_date]
        if len(valid_dates) > 0:
            return float(close.loc[valid_dates[-1]])
        else:
            # If no date before, get first available
            return float(close.iloc[0])
            
    except Exception as e:
        print(f"Error fetching {date_str}: {e}")
        return None


def main():
    print("Fetching ALLE historical prices for quarter ends...\n")
    
    results = []
    
    for quarter, date_str in QUARTER_ENDS.items():
        year = quarter.split('-')[0]
        shares = SHARES_OUTSTANDING.get(year, 87.0)
        
        price = get_quarter_end_price('ALLE', date_str)
        
        if price:
            market_cap = price * shares  # In millions
            results.append({
                'Quarter': quarter,
                'Date': date_str,
                'Price': price,
                'Shares_M': shares,
                'Market_Cap_M': market_cap
            })
            print(f"{quarter}: ${price:.2f} × {shares}M shares = ${market_cap:,.0f}M market cap")
        else:
            print(f"{quarter}: Could not fetch price")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print as Python dict format for easy copy-paste
    print("\n" + "="*60)
    print("COPY THIS INTO YOUR CODE:")
    print("="*60 + "\n")
    
    print("ALLE_CAPITAL_STRUCTURE = {")
    for _, row in df.iterrows():
        q = row['Quarter']
        mc = row['Market_Cap_M']
        print(f"    '{q}': {{'total_debt': XXXX, 'market_cap': {mc:.0f}, 'tax_rate': 0.XX}},")
    print("}")
    
    # Save to CSV
    df.to_csv('alle_market_cap_history.csv', index=False)
    print("\n✓ Saved to alle_market_cap_history.csv")
    
    return df


if __name__ == "__main__":
    df = main()