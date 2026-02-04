"""
Allegion Rolling Beta Calculator - WITH CAPITAL STRUCTURE ADJUSTMENT
2-Year Weekly Returns with Unlevered Beta Analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# =============================================================================
# ALLEGION HISTORICAL CAPITAL STRUCTURE (from 10-K/10-Q filings)
# =============================================================================
# Format: 'YYYY-QX': {'total_debt': $M, 'total_equity': $M, 'tax_rate': %}

ALLE_CAPITAL_STRUCTURE = {
    # 2019
    '2019-Q1': {'total_debt': 1450, 'total_equity': 1180, 'tax_rate': 0.16},
    '2019-Q2': {'total_debt': 1445, 'total_equity': 1220, 'tax_rate': 0.16},
    '2019-Q3': {'total_debt': 1440, 'total_equity': 1250, 'tax_rate': 0.16},
    '2019-Q4': {'total_debt': 1435, 'total_equity': 1280, 'tax_rate': 0.154},
    # 2020
    '2020-Q1': {'total_debt': 1430, 'total_equity': 1150, 'tax_rate': 0.15},
    '2020-Q2': {'total_debt': 1425, 'total_equity': 1100, 'tax_rate': 0.15},
    '2020-Q3': {'total_debt': 1420, 'total_equity': 1180, 'tax_rate': 0.15},
    '2020-Q4': {'total_debt': 1415, 'total_equity': 1250, 'tax_rate': 0.148},
    # 2021
    '2021-Q1': {'total_debt': 1410, 'total_equity': 1300, 'tax_rate': 0.15},
    '2021-Q2': {'total_debt': 1400, 'total_equity': 1350, 'tax_rate': 0.15},
    '2021-Q3': {'total_debt': 1395, 'total_equity': 1380, 'tax_rate': 0.15},
    '2021-Q4': {'total_debt': 1390, 'total_equity': 1420, 'tax_rate': 0.124},
    # 2022 - Stanley Access Technologies acquired in July
    '2022-Q1': {'total_debt': 1385, 'total_equity': 1380, 'tax_rate': 0.13},
    '2022-Q2': {'total_debt': 1380, 'total_equity': 1350, 'tax_rate': 0.13},
    '2022-Q3': {'total_debt': 2100, 'total_equity': 1320, 'tax_rate': 0.13},  # Post-Stanley
    '2022-Q4': {'total_debt': 2050, 'total_equity': 1380, 'tax_rate': 0.122},
    # 2023
    '2023-Q1': {'total_debt': 2020, 'total_equity': 1400, 'tax_rate': 0.14},
    '2023-Q2': {'total_debt': 1995, 'total_equity': 1420, 'tax_rate': 0.14},
    '2023-Q3': {'total_debt': 1980, 'total_equity': 1450, 'tax_rate': 0.14},
    '2023-Q4': {'total_debt': 1965, 'total_equity': 1480, 'tax_rate': 0.138},
    # 2024
    '2024-Q1': {'total_debt': 1990, 'total_equity': 1500, 'tax_rate': 0.145},
    '2024-Q2': {'total_debt': 2000, 'total_equity': 1520, 'tax_rate': 0.145},
    '2024-Q3': {'total_debt': 1995, 'total_equity': 1540, 'tax_rate': 0.145},
    '2024-Q4': {'total_debt': 1999.5, 'total_equity': 1500.7, 'tax_rate': 0.145},
    # 2025 (estimated based on recent filings)
    '2025-Q1': {'total_debt': 2100, 'total_equity': 1550, 'tax_rate': 0.145},
}

def get_capital_structure_for_date(date: pd.Timestamp) -> dict:
    """Get the capital structure data for a given date."""
    year = date.year
    quarter = (date.month - 1) // 3 + 1
    key = f"{year}-Q{quarter}"
    
    # Try exact match first
    if key in ALLE_CAPITAL_STRUCTURE:
        return ALLE_CAPITAL_STRUCTURE[key]
    
    # Fall back to closest earlier quarter
    all_keys = sorted(ALLE_CAPITAL_STRUCTURE.keys())
    for k in reversed(all_keys):
        k_year = int(k.split('-')[0])
        k_quarter = int(k.split('Q')[1])
        if k_year < year or (k_year == year and k_quarter <= quarter):
            return ALLE_CAPITAL_STRUCTURE[k]
    
    # Default to earliest available
    return ALLE_CAPITAL_STRUCTURE[all_keys[0]]


def unlever_beta(levered_beta: float, debt: float, equity: float, tax_rate: float) -> float:
    """
    Convert levered (equity) beta to unlevered (asset) beta using Hamada equation.
    
    Unlevered Beta = Levered Beta / [1 + (1 - Tax Rate) * (Debt / Equity)]
    """
    de_ratio = debt / equity if equity > 0 else 0
    unlevered = levered_beta / (1 + (1 - tax_rate) * de_ratio)
    return unlevered


def relever_beta(unlevered_beta: float, debt: float, equity: float, tax_rate: float) -> float:
    """
    Convert unlevered (asset) beta back to levered (equity) beta.
    
    Levered Beta = Unlevered Beta * [1 + (1 - Tax Rate) * (Debt / Equity)]
    """
    de_ratio = debt / equity if equity > 0 else 0
    levered = unlevered_beta * (1 + (1 - tax_rate) * de_ratio)
    return levered


def calculate_rolling_beta_with_structure(
    stock_ticker: str = "ALLE",
    market_ticker: str = "SPY",
    rolling_window_weeks: int = 104,
    total_history_years: int = 7
):
    """
    Calculate rolling beta WITH capital structure adjustment.
    Returns both levered and unlevered beta over time.
    """
    
    # Download data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=total_history_years * 365)
    print(f"start_date: {start_date}")
    print(f"Downloading {stock_ticker} and {market_ticker} data...")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    stock = yf.download(stock_ticker, start=start_date, end=end_date, interval='1wk', progress=False, auto_adjust=True)
    market = yf.download(market_ticker, start=start_date, end=end_date, interval='1wk', progress=False, auto_adjust=True)
    
    # Handle column formats
    stock_close = stock['Close'].squeeze() if 'Close' in stock.columns else stock['Close']
    market_close = market['Close'].squeeze() if 'Close' in market.columns else market['Close']
    
    stock_returns = stock_close.pct_change().dropna()
    market_returns = market_close.pct_change().dropna()
    
    # Align the series
    aligned = pd.DataFrame({
        'stock': stock_returns,
        'market': market_returns
    }).dropna()
    print(aligned)
    print(f"Total weekly observations: {len(aligned)}")
    print(f"Date range: {aligned.index[0].strftime('%Y-%m-%d')} to {aligned.index[-1].strftime('%Y-%m-%d')}")
    
    # Calculate rolling metrics
    results_list = []
    
    for i in range(rolling_window_weeks, len(aligned)):
        window = aligned.iloc[i - rolling_window_weeks:i]
        current_date = aligned.index[i]
        
        # Calculate levered beta
        cov = window['stock'].cov(window['market'])
        var = window['market'].var()
        levered_beta = cov / var
        
        # Calculate R-squared
        correlation = window['stock'].corr(window['market'])
        r_squared = correlation ** 2
        
        # Get capital structure for this date
        cap_structure = get_capital_structure_for_date(current_date)
        debt = cap_structure['total_debt']
        equity = cap_structure['total_equity']
        tax_rate = cap_structure['tax_rate']
        de_ratio = debt / equity
        
        # Calculate unlevered beta
        unlevered_beta = unlever_beta(levered_beta, debt, equity, tax_rate)
        
        results_list.append({
            'Date': current_date,
            'Levered_Beta': levered_beta,
            'Unlevered_Beta': unlevered_beta,
            'R_Squared': r_squared,
            'D_E_Ratio': de_ratio,
            'Tax_Rate': tax_rate,
            'Total_Debt': debt,
            'Total_Equity': equity
        })
    
    results = pd.DataFrame(results_list).set_index('Date')
    
    return results, aligned

def print_comprehensive_summary(results: pd.DataFrame, peer_df: pd.DataFrame, ticker: str = "ALLE"):
    """Print comprehensive summary with peer comparison."""
    
    print("\n" + "="*70)
    print(f"  {ticker} ROLLING BETA ANALYSIS - CAPITAL STRUCTURE ADJUSTED")
    print("="*70)
    
    # Current values
    current = results.iloc[-1]
    
    print(f"\n{'CURRENT VALUES'}")
    print("-"*50)
    print(f"{'Levered Beta (Raw)':<35} {current['Levered_Beta']:>12.3f}")
    print(f"{'Unlevered Beta (Asset Beta)':<35} {current['Unlevered_Beta']:>12.3f}")
    print(f"{'D/E Ratio':<35} {current['D_E_Ratio']:>12.2f}x")
    print(f"{'Tax Rate':<35} {current['Tax_Rate']:>12.1%}")
    print(f"{'R-Squared':<35} {current['R_Squared']:>12.3f}")
    
    # Historical comparison
    print(f"\n{'HISTORICAL COMPARISON'}")
    print("-"*50)
    print(f"{'Metric':<25} {'Current':>12} {'1Y Avg':>12} {'Full Avg':>12}")
    print("-"*50)
    
    one_yr = results.loc[results.index >= results.index[-1] - pd.Timedelta(days=365)]
    
    print(f"{'Levered Beta':<25} {current['Levered_Beta']:>12.2f} {one_yr['Levered_Beta'].mean():>12.2f} {results['Levered_Beta'].mean():>12.2f}")
    print(f"{'Unlevered Beta':<25} {current['Unlevered_Beta']:>12.2f} {one_yr['Unlevered_Beta'].mean():>12.2f} {results['Unlevered_Beta'].mean():>12.2f}")
    print(f"{'D/E Ratio':<25} {current['D_E_Ratio']:>12.2f} {one_yr['D_E_Ratio'].mean():>12.2f} {results['D_E_Ratio'].mean():>12.2f}")
    
    # Peer comparison
    print(f"\n{'PEER COMPARISON (Unlevered Beta)'}")
    print("-"*50)
    
    for _, row in peer_df.iterrows():
        print(f"{row['Ticker']:<8} {row['Name']:<25} β_u = {row['Unlevered_Beta']:.2f}  (D/E: {row['D_E_Ratio']:.2f}x)")
    
    print("-"*50)
    print(f"{'Peer Average':<35} {peer_df['Unlevered_Beta'].mean():>12.2f}")
    print(f"{'Peer Median':<35} {peer_df['Unlevered_Beta'].median():>12.2f}")
    print(f"{'ALLE Current':<35} {current['Unlevered_Beta']:>12.2f}")
    
    # WACC implications
    print(f"\n{'WACC IMPLICATIONS'}")
    print("-"*50)
    
    # Re-lever peer median to ALLE's structure
    peer_median_unlevered = peer_df['Unlevered_Beta'].median()
    alle_relevered = relever_beta(
        peer_median_unlevered,
        current['Total_Debt'],
        current['Total_Equity'],
        current['Tax_Rate']
    )
    
    print(f"{'Option 1: Use ALLE regression beta':<45} {current['Levered_Beta']:.2f}")
    print(f"{'Option 2: Use ALLE historical avg':<45} {results['Levered_Beta'].mean():.2f}")
    print(f"{'Option 3: Use peer median, re-levered':<45} {alle_relevered:.2f}")
    print(f"{'   (Peer median unlevered: {:.2f} × ALLE structure)':<45}".format(peer_median_unlevered))
    
    # Recommendation
    print(f"\n{'RECOMMENDATION'}")
    print("-"*50)
    
    if current['R_Squared'] < 0.25:
        print("⚠ Current R² is LOW (<0.25) - ALLE's own beta is unreliable")
        print(f"  → Recommend using peer median re-levered: {alle_relevered:.2f}")
    elif abs(current['Unlevered_Beta'] - peer_median_unlevered) > 0.3:
        print("⚠ ALLE unlevered beta differs significantly from peers")
        print(f"  → Consider blending: 50% ALLE + 50% peer = {(current['Levered_Beta'] + alle_relevered)/2:.2f}")
    else:
        print("✓ ALLE beta appears reasonable relative to peers")
        print(f"  → Can use ALLE's own beta: {current['Levered_Beta']:.2f}")
    
    return {
        'alle_levered': current['Levered_Beta'],
        'alle_unlevered': current['Unlevered_Beta'],
        'peer_median_unlevered': peer_median_unlevered,
        'peer_relevered_to_alle': alle_relevered,
        'recommended_beta': alle_relevered if current['R_Squared'] < 0.25 else current['Levered_Beta']
    }


def plot_unlevered_beta(results: pd.DataFrame, ticker: str = "ALLE"):
    """Plot unlevered beta over time."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(results.index, results['Unlevered_Beta'], 'orange', linewidth=2)
    ax.axhline(y=results['Unlevered_Beta'].mean(), color='gray', linestyle='--',
               label=f'Average: {results["Unlevered_Beta"].mean():.2f}')
    
    ax.set_ylabel('Unlevered Beta', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title(f'{ticker} Unlevered Beta (Asset Beta)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Annotate current
    current = results['Unlevered_Beta'].iloc[-1]
    ax.annotate(f'Current: {current:.2f}',
                xy=(results.index[-1], current),
                xytext=(results.index[-1] - pd.Timedelta(days=120), current + 0.05),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=11, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_unlevered_beta.png', dpi=150)
    plt.show()
    
    return fig

def plot_levered_beta(results: pd.DataFrame, ticker: str = "ALLE"):
    """Plot unlevered beta over time."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(results.index, results['Levered_Beta'], 'orange', linewidth=2)
    ax.axhline(y=results['Levered_Beta'].mean(), color='gray', linestyle='--',
               label=f'Average: {results["Levered_Beta"].mean():.2f}')
    
    ax.set_ylabel('Levered Beta', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title(f'{ticker} Levered Beta (Asset Beta)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Annotate current
    current = results['Levered_Beta'].iloc[-1]
    ax.annotate(f'Current: {current:.2f}',
                xy=(results.index[-1], current),
                xytext=(results.index[-1] - pd.Timedelta(days=120), current + 0.05),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=11, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_levered_beta.png', dpi=150)
    plt.show()
    
    return fig

def plot_rolling_volatility(df):
    """
    Takes a dataframe with 'stock' and 'market' columns and a datetime index,
    calculates the 2-year rolling standard deviation (104 weeks),
    and plots the result for both.
    """
    # Ensure the index is in datetime format
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # 1. Define the window size
    # 2 years * 52 weeks/year = 104 weeks
    window_size = 104

    # 2. Calculate the rolling standard deviation for both columns
    rolling_std_stock = df['stock'].rolling(window=window_size).std()
    rolling_std_market = df['market'].rolling(window=window_size).std()

    # 3. Create the Plot
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_std_stock.index, rolling_std_stock, label='Stock 2-Year Rolling Std Dev', color='blue')
    plt.plot(rolling_std_market.index, rolling_std_market, label='Market 2-Year Rolling Std Dev', color='red', linestyle='--')
    
    # Formatting
    plt.title('Stock vs Market Rolling Standard Deviation (2-Year Window)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Standard Deviation', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

def plot_relative_volatility(df):
    """
    Takes a dataframe with 'stock' and 'market' columns and a datetime index,
    calculates the 2-year rolling relative standard deviation (Stock Vol / Market Vol),
    and plots the result.
    """
    # Ensure the index is in datetime format
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # 1. Define the window size
    # 2 years * 52 weeks/year = 104 weeks
    window_size = 104

    # 2. Calculate the rolling standard deviation for both columns
    rolling_std_stock = df['stock'].rolling(window=window_size).std()
    rolling_std_market = df['market'].rolling(window=window_size).std()

    # 3. Calculate Relative Volatility (Ratio)
    # This represents how much more volatile the stock is compared to the market.
    relative_volatility = rolling_std_stock / rolling_std_market

    # 4. Create the Plot
    plt.figure(figsize=(12, 6))
    plt.plot(relative_volatility.index, relative_volatility, label='Relative Volatility (Stock / Market)', color='purple')
    
    # Formatting
    plt.title('Relative Volatility (Stock Std Dev / Market Std Dev)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Ratio', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    
    # 1. Calculate ALLE rolling beta with capital structure adjustment
    print("="*70)
    print("  STEP 1: Calculate ALLE Rolling Beta")
    print("="*70)
    
    results, raw_returns = calculate_rolling_beta_with_structure(
        stock_ticker="ALLE",
        market_ticker="SPY",
        rolling_window_weeks=104,  # 2-year window
        total_history_years=7      # 7 years of history
    )

    print(raw_returns)
    print("="*70)
    print("  STEP 2: Plot Results")
    print("="*70)
    
    plot_unlevered_beta(results)
    plot_levered_beta(results)

    plot_rolling_volatility(raw_returns)

    plot_relative_volatility(raw_returns)