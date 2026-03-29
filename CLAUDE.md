# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Python 3.13, managed with `uv`
- Dependencies: `numpy`, `pandas`, `yfinance`, `matplotlib`, `openpyxl`

```bash
uv run python <script.py>   # run any script
uv add <package>            # add a dependency
```

## What This Project Is

A collection of standalone analysis scripts for calculating WACC (Weighted Average Cost of Capital) inputs for **Allegion PLC (ALLE)**. There is no web app or CLI entrypoint — each script is run directly.

## Script Inventory

| Script | Purpose |
|--------|---------|
| `beta/allegion_beta_5yr_monthly.py` | Point-in-time beta: 5-year monthly regression + peer comparison, outputs Excel + charts |
| `beta/allegion_beta_2yr_weekly.py` | Same as above but 2-year weekly returns |
| `beta/rolling_beta.py` | Rolling 2-year weekly levered beta over 7 years of history, outputs CSV + chart |
| `beta/rolling_unlevered_beta.py` | Rolling beta WITH capital structure adjustment (levered + unlevered over time) |
| `std_dev.py` | Rolling volatility comparison: ALLE vs SPY, ASAZY, DOKA.SW — absolute and relative |
| `MV_equity/market_cap.py` | Entry point that calls `src/wacc/beta/marketCap.py` to build a daily market cap + capital structure DataFrame, saves to `data/marketCap.csv` |
| `stock_chart_simple.py` | Downloads ALLE, S&P 500, and XLI, normalizes to 100, saves to Excel |

## Architecture

### Two parallel code structures

**`src/wacc/`** — the only proper Python package (installed via `pyproject.toml`). Currently contains one module:
- `src/wacc/beta/marketCap.py`: `get_wacc_inputs()` merges daily yfinance market cap data with the manually-maintained `ALLE_CAPITAL_STRUCTURE` quarterly dict (debt + tax rate from 10-K/10-Q filings). Imported by `MV_equity/market_cap.py` via `from wacc.beta import marketCap`.

**`beta/`** — standalone scripts (not installable). Each script is self-contained with its own data, functions, and `main()`.

### Capital structure data

The quarterly ALLE capital structure (debt, tax rate, sometimes equity) is hardcoded in two places:
- `src/wacc/beta/marketCap.py` — `ALLE_CAPITAL_STRUCTURE` dict (debt + tax, 2019–2025)
- `beta/rolling_unlevered_beta.py` — duplicate of same dict

When updating capital structure data (e.g., after a new 10-K), update **both** locations. The 2024 10-K values (debt: $1,999.5M, equity: $1,500.7M, tax: 14.5%) are the current baseline used in `beta/allegion_beta_*.py`.

### Beta methodology

All scripts use the **Hamada equation** for levering/unlevering:
- Unlever: `β_U = β_L / [1 + (1-T) × D/E]`
- Relever: `β_L = β_U × [1 + (1-T) × D/E]`

The peer set differs between scripts — `allegion_beta_5yr_monthly.py` uses 7 peers (FBIN, MAS, SWK, JCI, CARR, REZI, ASAZY); `allegion_beta_2yr_weekly.py` uses 4 tighter comps (ASAZY, DOKA, JCI, FBIN).

### yfinance column handling

yfinance returns a `MultiIndex` when downloading multiple tickers. Scripts use `.squeeze()` or level-0 extraction to flatten to a plain Series. The `rolling_unlevered_beta.py` uses `auto_adjust=True`; older scripts use `Adj Close` fallback logic.
