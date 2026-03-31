"""
ALLE vs SPY Returns from 2013
Downloads historical data for Allegion (ALLE) and S&P 500 (SPY)
Calculates and plots daily and cumulative returns
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ============================================
# CONFIGURATION
# ============================================
TICKERS = {
    "ALLE": "ALLE",       # Allegion PLC
    "SPY": "SPY"          # S&P 500 ETF
}
START_DATE = "2013-01-01"
END_DATE = "2026-03-30"
# ============================================

def calculate_returns(prices):
    """Calculate daily and cumulative returns"""
    daily_returns = prices.pct_change() * 100  # daily % returns
    cumulative_returns = (1 + daily_returns / 100).cumprod() - 1  # cumulative return
    cumulative_returns = cumulative_returns * 100  # convert to %
    return daily_returns, cumulative_returns

# Download data
data = {}
all_dates = []
for name, ticker in TICKERS.items():
    print(f"Downloading {name} ({ticker}) data from {START_DATE} to {END_DATE}...")
    stock = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)

    if stock.empty:
        print(f"  No data found for {ticker}")
        continue

    print(f"  Downloaded {len(stock)} data points")

    # Flatten multi-index if needed
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = stock.columns.get_level_values(0)

    # Use Adj Close (or Close as fallback)
    if 'Adj Close' in stock.columns:
        prices = stock['Adj Close'].dropna()
    else:
        prices = stock['Close'].dropna()
    data[name] = prices
    all_dates.append(prices.index)

# Align to same start date
if all_dates:
    max_start_date = max(d[0] for d in all_dates)
    for name in data.keys():
        data[name] = data[name][data[name].index >= max_start_date]
    print(f"\nAligned to common start date: {max_start_date.strftime('%Y-%m-%d')}")

# Calculate returns
returns_data = {}
for name, prices in data.items():
    daily_ret, cumul_ret = calculate_returns(prices)
    returns_data[name] = {
        'daily': daily_ret,
        'cumulative': cumul_ret,
        'prices': prices
    }

# Create figure with GridSpec layout: 2 cols, left col has 2 rows
fig = plt.figure(figsize=(14, 7))
fig.suptitle('ALLE vs SPY Performance Analysis (2013-2026)', fontsize=14, fontweight='bold')

gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[2, 1], height_ratios=[1, 1], hspace=0.3, wspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])  # Top left: Price
ax2 = fig.add_subplot(gs[1, 0])  # Bottom left: Cumulative returns
ax3 = fig.add_subplot(gs[:, 1])  # Right: Daily returns (spans both rows)

# Define colors (inverted order)
colors = {'SPY': '#1f77b4', 'ALLE': '#ff7f0e'}  # SPY blue, ALLE orange

# Plot 1: Price chart (normalized to 100)
for name in ['SPY', 'ALLE']:
    if name in data:
        prices = data[name]
        normalized = (prices / prices.iloc[0]) * 100
        ax1.plot(normalized.index, normalized.values, label=name, linewidth=2, color=colors[name])
ax1.set_title('Normalized Price (Start = 100)', fontweight='bold')
ax1.set_ylabel('Index Value')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2: Cumulative returns
for name in ['SPY', 'ALLE']:
    if name in returns_data:
        ret_data = returns_data[name]
        ax2.plot(ret_data['cumulative'].index, ret_data['cumulative'].values,
                 label=name, linewidth=2, color=colors[name])
ax2.set_title('Cumulative Returns (%)', fontweight='bold')
ax2.set_ylabel('Return (%)')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# Plot 3: Daily returns distribution
for name in ['SPY', 'ALLE']:
    if name in returns_data:
        ret_data = returns_data[name]
        ax3.hist(ret_data['daily'].dropna(), bins=100, alpha=0.6, label=name, color=colors[name])
ax3.set_title('Daily Returns\nDistribution (%)', fontweight='bold')
ax3.set_xlabel('Daily Return (%)')
ax3.set_ylabel('Frequency')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Save plot
output_dir = os.path.dirname(os.path.abspath(__file__))
plot_path = os.path.join(output_dir, 'alle_spy_returns_2013.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {plot_path}")

# Print statistics
print("\n" + "="*60)
print("STATISTICS (2013-2026)")
print("="*60)
for name, ret_data in returns_data.items():
    prices = data[name]
    cumul = ret_data['cumulative'].iloc[-1]
    daily = ret_data['daily'].dropna()

    print(f"\n{name}:")
    print(f"  Starting Price: ${prices.iloc[0]:.2f}")
    print(f"  Ending Price: ${prices.iloc[-1]:.2f}")
    print(f"  Total Return: {cumul:.2f}%")
    print(f"  Annualized Return: {(cumul / 13):.2f}%")  # ~13 years
    print(f"  Daily Avg Return: {daily.mean():.4f}%")
    print(f"  Daily Std Dev: {daily.std():.4f}%")
    print(f"  Min Daily Return: {daily.min():.2f}%")
    print(f"  Max Daily Return: {daily.max():.2f}%")

plt.show()
