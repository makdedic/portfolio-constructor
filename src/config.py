"""Every constant used across the pipeline lives here — nothing is hardcoded
in individual modules, so any assumption (dates, costs, thresholds) can be
found and changed in one place.
"""

from datetime import date, timedelta
from pathlib import Path

# --- Ticker universe -------------------------------------------------------
# Kept as the default (rather than the full S&P 500) for two reasons: dev/
# notebook iteration speed and Streamlit dashboard responsiveness both stay
# fast, and config.py must never require network I/O at import time (nearly
# every module transitively imports it) — a live-fetched S&P 500 list can't
# live at module scope regardless of which universe is "default". The full
# list is available via ingest.load_or_fetch_sp500_tickers(), used explicitly
# (e.g. run_backtest(..., tickers=sp500_tickers)) where it's actually wanted.
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
BACKTEST_START_DATE = date(2015, 1, 1)


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
TRANSACTION_COST_BPS = 10  # flat per-rebalance rate, applied to traded notional (turnover)

# A flat rate alone ignores market impact: the effect of your own order
# moving the price against you, which grows with trade size relative to
# how much of that stock normally trades. Real market impact research
# finds this cost scales roughly with the square root of trade size, not
# linearly with it - so a rebalance that trades twice as much costs more
# per dollar traded, not just proportionally more overall. This surcharge
# rate (in bps) is multiplied by sqrt(turnover) and added to the flat rate
# above - see backtest._transaction_cost.
MARKET_IMPACT_BPS_PER_SQRT_TURNOVER = 10

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

# Flat fallback risk-free rate, used when no FRED series is supplied (e.g.
# synthetic-data unit tests, or a quick run without network access). Real
# runs use the actual daily FRED 3-month T-bill history instead — see
# RISK_FREE_RATE_FRED_SERIES and src/data/ingest.py.
RISK_FREE_RATE_ANNUAL = 0.02
RISK_FREE_RATE_FRED_SERIES = "DTB3"  # 3-month T-bill, secondary market rate

# --- S&P 500 universe (full, optional) ----------------------------------------
SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Retry-with-backoff for yf.download() (prices and dividends both route
# through it - see src/data/ingest.py's _download_with_retry) if Yahoo
# rate-limits the request. Not a guarantee if the request volume itself is
# what's triggering it - see that function's docstring for what was
# actually verified. yf.download() already defaults to a 10s per-request
# timeout, so no separate timeout setting is needed here.
YFINANCE_DOWNLOAD_MAX_RETRIES = 3
YFINANCE_DOWNLOAD_BACKOFF_BASE_SECONDS = 2

# --- LightGBM ranker -------------------------------------------------------
# The training panel is only ~1,000-1,300 rows (~39 tickers x ~30 monthly
# snapshots), so these stay conservative to avoid overfitting. LightGBM's
# own default min_child_samples (20) is high enough relative to this row
# count that it can produce a near-constant, unhelpful predictor — lower it.
LGBM_N_ESTIMATORS = 50
LGBM_MAX_DEPTH = 3
LGBM_LEARNING_RATE = 0.05
LGBM_MIN_CHILD_SAMPLES = 10

# --- Storage -------------------------------------------------------------
# Anchored to the project root (not a relative path) so the cache always
# lands in the same place regardless of which directory a script, test, or
# notebook happens to be run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
DUCKDB_PATH = CACHE_DIR / "portfolio.duckdb"

# Committed to the repo (unlike CACHE_DIR) so a freshly deployed app starts
# already warm - see storage.get_connection / scripts/build_seed.py. Built
# once locally, where fetching is reliable, and shipped with the app rather
# than fetched live on Streamlit Cloud's shared IP, which was confirmed
# directly to be rate-limited independent of our own request pattern.
SEED_DIR = PROJECT_ROOT / "data" / "seed"
