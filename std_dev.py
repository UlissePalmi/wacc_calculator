import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_all_returns(tickers=["ALLE", "SPY", "ASAZY", "DOKA.SW"]):
    """
    Fetches max history for a LIST of tickers, aligns dates (handling different 
    market holidays), and calculates weekly log returns for all of them.
    
    Returns:
        pd.DataFrame: A dataframe where columns are the tickers (e.g. 'ALLE', 'SPY'...)
                      containing the log returns.
    """
    print(f"Downloading max history for: {tickers}...")
    
    # 1. Download Max History for all tickers at once
    # group_by='ticker' ensures the columns are easy to access
    data = yf.download(tickers, period="max", interval='1wk', progress=False, auto_adjust=True)
    
    # 2. Extract 'Close' Prices
    # yfinance returns a MultiIndex (Price Type, Ticker) -> We just want 'Close'
    if isinstance(data.columns, pd.MultiIndex):
        df_prices = data['Close']
    else:
        # Fallback if only 1 ticker is requested (it returns a flat index)
        df_prices = data['Close'].to_frame(name=tickers[0])

    # 3. Clean and Align Data
    # 'ffill()' stands for Forward Fill. If the Swiss market is closed for a holiday 
    # but US is open, we carry forward the last Swiss price so we don't drop the whole week.
    df_prices = df_prices.ffill().dropna()

    # 4. Calculate Log Returns for ALL columns
    # Formula: ln(Price_t / Price_t-1)
    df_returns = np.log(df_prices / df_prices.shift(1))
    
    # Drop the first row (NaN) created by the shift
    df_clean = df_returns.dropna()
    
    return df_clean

def plot_rolling_std_dev(df, window_weeks=104, annualize=False):
    """
    Takes a dataframe of returns (multiple columns), calculates rolling std dev
    for ALL of them, and plots them on a single comparison chart.
    
    Parameters:
        df: Dataframe containing log returns for multiple tickers.
        window_weeks: Rolling window size (default 104).
        annualize: If True, multiplies by sqrt(52) for annualized volatility.
    """
    # 1. Calculate Rolling Standard Deviation for the entire dataframe
    rolling_std = df.rolling(window=window_weeks).std()
    
    # Drop the initial NaN period
    rolling_std = rolling_std.dropna()
    
    # (Optional) Annualize
    if annualize:
        rolling_std = rolling_std * np.sqrt(52)
        ylabel = "Annualized Volatility"
    else:
        ylabel = "Weekly Standard Deviation"

    # 2. Setup Plot
    plt.figure(figsize=(14, 7))
    
    # Define Professional Colors
    colors = {
        'ALLE': '#003366',    # Navy Blue (Main Focus)
        'SPY': '#d9534f',     # Muted Red (Market)
        'ASAZY': '#5cb85c',   # Green (Peer 1)
        'DOKA.SW': '#f0ad4e', # Orange (Peer 2)
        'DRRKF': '#f0ad4e'    # Orange (Alt Peer)
    }

    # 3. Plot Each Ticker
    for ticker in rolling_std.columns:
        # Determine style based on ticker
        color = colors.get(ticker, None) # Use mapped color or default
        linewidth = 2.5 if ticker == 'ALLE' else 1.5 # Make ALLE thicker
        linestyle = '--' if ticker == 'SPY' else '-' # Make Market dashed
        alpha = 0.8 if ticker == 'SPY' else 1.0      # Make Market slightly transparent
        
        plt.plot(rolling_std.index, rolling_std[ticker], 
                 label=f"{ticker} Volatility", 
                 color=color, 
                 linewidth=linewidth, 
                 linestyle=linestyle,
                 alpha=alpha)
    
    # 4. Formatting
    plt.title(f'Volatility Comparison ({window_weeks}-Week Rolling)', fontsize=16)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel('Year', fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def plot_relative_std_dev(df, window_weeks=104, start_date='2019-01-01'):
    """
    Plots the Relative Volatility (Stock / Market) starting specifically from 2019.
    """
    # 1. Calculate Rolling Standard Deviations (using FULL history first)
    # We do the math before filtering so 2019 starts with valid data immediately
    stock_roll_std = df['ALLE'].rolling(window=window_weeks).std()
    market_roll_std = df['SPY'].rolling(window=window_weeks).std()
    
    # 2. Calculate the Ratio
    relative_vol = stock_roll_std / market_roll_std
    
    # 3. Slice the data to only keep 2019 onwards
    # This hides the older data without breaking the calculation
    relative_vol = relative_vol.loc[start_date:]
    
    # 4. Plotting
    plt.figure(figsize=(12, 6))
    
    # Main Line
    plt.plot(relative_vol.index, relative_vol, 
             label='Relative Volatility (ALLE / SPY)', color='purple', linewidth=2)
    
    # Average Line (Calculated only based on the displayed period)
    avg_rel_vol = relative_vol.mean()
    plt.axhline(avg_rel_vol, color='green', linestyle='--', alpha=0.7, 
                label=f'Average Since 2019 ({avg_rel_vol:.2f}x)')
    
    plt.title(f'Relative Volatility Ratio (Since {start_date})', fontsize=14)
    plt.ylabel('Ratio (Stock Vol / Market Vol)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# --- Full Execution Flow ---

# 1. Get the Data (Max history)
df_all = get_all_returns(["ALLE", "SPY", "ASAZY", "DOKA.SW"])

# 2. Plot Absolute Volatility (The "Decoupling" Chart)
plot_rolling_std_dev(df_all, window_weeks=104, annualize=False)

# 3. Plot Relative Volatility (The Ratio Chart)
plot_relative_std_dev(df_all, window_weeks=104)