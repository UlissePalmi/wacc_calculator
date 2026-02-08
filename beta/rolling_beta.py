"""
Allegion Rolling Beta Calculator
2-Year Weekly Returns with Rolling Window Analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def calculate_rolling_beta(stock_ticker: str = "ALLE", 
                          market_ticker: str = "SPY",
                          rolling_window_weeks: int = 104,  # 2 years
                          total_history_years: int = 7):    # Get 7 years to see evolution
    """
    Calculate rolling 2-year weekly beta for a stock.
    
    Parameters:
    - rolling_window_weeks: 104 weeks = 2 years
    - total_history_years: How far back to pull data
    """
    
    # Download data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=total_history_years * 365)
    
    print(f"Downloading {stock_ticker} and {market_ticker} data...")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    stock = yf.download(stock_ticker, start=start_date, end=end_date, interval='1wk', progress=False)
    market = yf.download(market_ticker, start=start_date, end=end_date, interval='1wk', progress=False)
    
    # Calculate weekly returns
    stock_returns = stock['Close'].squeeze().pct_change().dropna()
    market_returns = market['Close'].squeeze().pct_change().dropna()

    # Align the series
    aligned = pd.DataFrame({
        'stock': stock_returns,
        'market': market_returns
    }).dropna()
    
    print(f"Total weekly observations: {len(aligned)}")
    print(f"Date range: {aligned.index[0].strftime('%Y-%m-%d')} to {aligned.index[-1].strftime('%Y-%m-%d')}")
    
    # Calculate rolling beta
    rolling_betas = []
    rolling_dates = []
    rolling_r_squared = []
    
    for i in range(rolling_window_weeks, len(aligned)):
        window = aligned.iloc[i - rolling_window_weeks:i]
        
        # Calculate beta using covariance method
        cov = window['stock'].cov(window['market'])
        var = window['market'].var()
        beta = cov / var
        
        # Calculate R-squared
        correlation = window['stock'].corr(window['market'])
        r_squared = correlation ** 2
        
        rolling_betas.append(beta)
        rolling_dates.append(aligned.index[i])
        rolling_r_squared.append(r_squared)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Date': rolling_dates,
        'Rolling_Beta': rolling_betas,
        'R_Squared': rolling_r_squared
    }).set_index('Date')
    
    return results, aligned


def plot_rolling_beta(results: pd.DataFrame, ticker: str = "ALLE"):
    """Create visualization of rolling beta over time."""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Rolling Beta
    ax1 = axes[0]
    ax1.plot(results.index, results['Rolling_Beta'], 'b-', linewidth=1.5, label='Rolling 2Y Beta')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Market β=1.0')
    ax1.axhline(y=results['Rolling_Beta'].mean(), color='green', linestyle='--', alpha=0.7, 
                label=f'Average: {results["Rolling_Beta"].mean():.2f}')
    
    # Add bands for context
    ax1.fill_between(results.index, 0.8, 1.2, alpha=0.1, color='gray', label='Typical range')
    
    ax1.set_ylabel('Beta', fontsize=12)
    ax1.set_title(f'{ticker} Rolling 2-Year Weekly Beta', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(results['Rolling_Beta'].max() * 1.1, 1.5))
    
    # Annotate current beta
    current_beta = results['Rolling_Beta'].iloc[-1]
    ax1.annotate(f'Current: {current_beta:.2f}', 
                xy=(results.index[-1], current_beta),
                xytext=(results.index[-1] - pd.Timedelta(days=180), current_beta + 0.15),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=11, color='red', fontweight='bold')
    
    # Plot 2: R-Squared (regression quality)
    ax2 = axes[1]
    ax2.plot(results.index, results['R_Squared'], 'purple', linewidth=1.5, alpha=0.7)
    ax2.fill_between(results.index, 0, results['R_Squared'], alpha=0.3, color='purple')
    ax2.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='R²=0.30 threshold')
    
    ax2.set_ylabel('R-Squared', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title('Regression Quality (R²) Over Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('allegion_rolling_beta.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


def print_summary_stats(results: pd.DataFrame, ticker: str = "ALLE"):
    """Print summary statistics for the rolling beta analysis."""
    
    print("\n" + "="*60)
    print(f"  {ticker} ROLLING BETA SUMMARY (2-Year Weekly)")
    print("="*60)
    
    print(f"\n{'Metric':<30} {'Value':>15}")
    print("-"*45)
    print(f"{'Current Beta':<30} {results['Rolling_Beta'].iloc[-1]:>15.3f}")
    print(f"{'Average Beta':<30} {results['Rolling_Beta'].mean():>15.3f}")
    print(f"{'Median Beta':<30} {results['Rolling_Beta'].median():>15.3f}")
    print(f"{'Min Beta':<30} {results['Rolling_Beta'].min():>15.3f}")
    print(f"{'Max Beta':<30} {results['Rolling_Beta'].max():>15.3f}")
    print(f"{'Std Dev':<30} {results['Rolling_Beta'].std():>15.3f}")
    
    print(f"\n{'Current R²':<30} {results['R_Squared'].iloc[-1]:>15.3f}")
    print(f"{'Average R²':<30} {results['R_Squared'].mean():>15.3f}")
    
    # Recent periods
    print("\n" + "-"*45)
    print("BETA BY PERIOD:")
    print("-"*45)
    
    # Last 6 months
    six_mo = results.loc[results.index >= results.index[-1] - pd.Timedelta(days=180), 'Rolling_Beta'].mean()
    one_yr = results.loc[results.index >= results.index[-1] - pd.Timedelta(days=365), 'Rolling_Beta'].mean()
    two_yr = results.loc[results.index >= results.index[-1] - pd.Timedelta(days=730), 'Rolling_Beta'].mean()
    
    print(f"{'Last 6 months avg':<30} {six_mo:>15.3f}")
    print(f"{'Last 1 year avg':<30} {one_yr:>15.3f}")
    print(f"{'Last 2 years avg':<30} {two_yr:>15.3f}")
    
    # Key events context
    print("\n" + "-"*45)
    print("KEY OBSERVATIONS:")
    print("-"*45)
    
    if results['Rolling_Beta'].iloc[-1] < 0.8:
        print("⚠ Current beta is LOW (<0.8) - stock trading defensively")
    elif results['Rolling_Beta'].iloc[-1] > 1.2:
        print("⚠ Current beta is HIGH (>1.2) - stock is volatile vs market")
    else:
        print("✓ Current beta is in normal range (0.8-1.2)")
    
    if results['R_Squared'].iloc[-1] < 0.3:
        print("⚠ Low R² - beta estimate may be unreliable")
    else:
        print("✓ R² is reasonable - beta estimate is meaningful")


def analyze_beta_vs_events(results: pd.DataFrame):
    """Analyze how beta changed around key events."""
    
    print("\n" + "="*60)
    print("  BETA AROUND KEY EVENTS")
    print("="*60)
    
    events = {
        '2020-03': 'COVID Crash',
        '2022-01': 'Rate Hike Cycle Begins', 
        '2022-07': 'Stanley Access Acquisition',
        '2023-03': 'Banking Crisis (SVB)',
        '2024-01': 'Post-Rate Peak',
    }
    
    for date_str, event_name in events.items():
        try:
            # Find closest date in results
            target_date = pd.Timestamp(date_str + '-01')
            mask = (results.index >= target_date) & (results.index < target_date + pd.Timedelta(days=60))
            if mask.any():
                beta_at_event = results.loc[mask, 'Rolling_Beta'].iloc[0]
                print(f"{event_name:<30} ({date_str}): β = {beta_at_event:.2f}")
        except:
            pass


if __name__ == "__main__":
    # Run the analysis
    results, raw_returns = calculate_rolling_beta(
        stock_ticker="ALLE",
        market_ticker="SPY", 
        rolling_window_weeks=104,  # 2-year window
        total_history_years=7       # 7 years of history
    )
    
    # Print summary
    print_summary_stats(results, "ALLE")
    
    # Analyze key events
    analyze_beta_vs_events(results)
    
    # Create visualization
    fig = plot_rolling_beta(results, "ALLE")
    
    # Export to CSV
    results.to_csv('allegion_rolling_beta.csv')
    print("\n✓ Results saved to allegion_rolling_beta.csv")
    print("✓ Chart saved to allegion_rolling_beta.png")