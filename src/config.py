"""Every constant used across the pipeline lives here — nothing is hardcoded
in individual modules, so any assumption (dates, costs, thresholds) can be
found and changed in one place.
"""

from datetime import date, timedelta
from pathlib import Path

# --- Ticker universe -------------------------------------------------------
# First use a subset of the S&P 500 instead of the full ~500 names.
# This keeps yfinance calls fast before we get the whole pipeline connected end to end.
# Swap TICKER_UNIVERSE for a full S&P 500 list later — nothing downstream needs to change.
DEV_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",       # tech
    "JPM", "BAC", "WFC", "GS",                                     # financials
    "JNJ", "PFE", "UNH", "MRK", "ABBV",                            # healthcare
    "PG", "KO", "PEP", "WMT", "HD", "MCD",                         # consumer
    "XOM", "CVX", "COP",                                           # energy
    "CAT", "BA", "GE", "HON", "UNP",                                # industrials
    "NEE", "DUK", "SO",                                            # utilities
    "T", "VZ",                                                     # telecom
    "CSCO", "INTC", "IBM", "ORCL",                                 # tech (again, large caps)
]
TICKER_UNIVERSE = DEV_TICKERS

# SPY, not the raw ^GSPC index, so the benchmark is priced with the same
# dividend-adjusted-close convention as every stock in TICKER_UNIVERSE.
BENCHMARK_TICKER = "SPY"

# --- Date ranges -------------------------------------------------------------
BACKTEST_START_DATE = date(2019, 1, 1)


def _last_completed_month_end() -> date:
    """The most recent calendar month-end that has fully elapsed."""
    first_of_this_month = date.today().replace(day=1)
    return first_of_this_month - timedelta(days=1)


BACKTEST_END_DATE = _last_completed_month_end()

TRAIN_WINDOW_YEARS = 3

# Price history must start well before the first rebalance date: the first
# rebalance's training window reaches back TRAIN_WINDOW_YEARS, and factors
# computed at the start of that window need their own lookback (up to 12
# months) before they're usable. One extra year of buffer covers that.
DATA_START_DATE = date(
    BACKTEST_START_DATE.year - (TRAIN_WINDOW_YEARS + 1),
    BACKTEST_START_DATE.month,
    BACKTEST_START_DATE.day,
)
DATA_END_DATE = date.today()

# --- Rebalancing and portfolio construction ----------------------------------
REBALANCE_FREQUENCY = "ME"  # month-end, pandas offset alias
TOP_N_HOLDINGS = 15  # stocks selected by the ranker before optimisation
MAX_WEIGHT_PER_STOCK = 0.15  # optimiser bound, avoids single-name concentration
TRANSACTION_COST_BPS = 10  # per rebalance, applied to traded notional (turnover)

# --- Alpha factor lookback windows -------------------------------------------
MOMENTUM_LOOKBACK_MONTHS = 12
MOMENTUM_SKIP_MONTHS = 1  # classic "12-1" momentum: skips the most recent
# month, which is prone to short-term price reversal rather than trend
LOW_VOL_LOOKBACK_MONTHS = 12
DIVIDEND_YIELD_LOOKBACK_MONTHS = 12

# --- Risk metrics -------------------------------------------------------------
VAR_CONFIDENCE_LEVELS = [0.95, 0.99]
MONTE_CARLO_SIMULATIONS = 10_000
RANDOM_SEED = 42
TRADING_DAYS_PER_YEAR = 252

# Placeholder flat risk-free rate. A later pass replaces this with the FRED
# 3-month T-bill series via pandas-datareader; using 0 magic numbers here
# would just move the same problem into ranker.py/metrics.py instead of
# solving it, so it stays a named constant until FRED ingestion lands.
RISK_FREE_RATE_ANNUAL = 0.02

# --- Storage -------------------------------------------------------------
# Anchored to the project root (not a relative path) so the cache always
# lands in the same place regardless of which directory a script, test, or
# notebook happens to be run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
