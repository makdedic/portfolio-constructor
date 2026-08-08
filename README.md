# Portfolio Constructor

An ML-driven portfolio construction and risk analysis system: alpha factors
→ stock ranking → mean-variance optimisation → walk-forward backtest → risk
metrics, with a Streamlit dashboard on top. Built as an interview-ready
project for asset management / fintech / bank data science roles — every
output is meant to be explainable line by line.

The project follows a set of rules that are never broken, in any pass:

1. **No lookahead bias** — every feature uses only data strictly before the
   date it's predicting for.
2. **Walk-forward only** — a rolling training window that moves forward
   through time; never train on full history.
3. **Transaction costs** — applied on every rebalance, reported gross and
   net so the cost drag itself is visible.
4. **Ledoit-Wolf shrinkage** — always used for the covariance matrix, never
   raw sample covariance.
5. **Two benchmarks** — every backtest compares against both an
   equal-weight portfolio and S&P 500 buy-and-hold.
6. **Full risk-metric set** — Sharpe, Sortino, max drawdown, VaR (95% and
   99%), CVaR, and VaR breaches; never raw returns alone.

## What's implemented

Built in two passes: Pass 1 was a thin vertical slice (every stage
connected end-to-end, on a small dev universe, to prove the six rules
above hold in full before adding breadth); Pass 2 added the remaining
depth — a second ranker and optimiser, real market data in place of
placeholders, the full S&P 500 as an available universe, orchestration,
and proper storage. Both passes are complete.

- **Ingestion** (`src/data/ingest.py`, `src/data/storage.py`) — daily
  prices + dividends via yfinance, the risk-free rate via FRED, S&P 500
  constituents scraped from Wikipedia — all cached in DuckDB
  (`src/data/storage.py`) with per-ticker incremental fetching, so
  overlapping requests (e.g. the dev universe and the full S&P 500) share
  cached data instead of refetching
- **Alpha factors** (`src/data/features.py`) — 12-1 month momentum,
  trailing 12-month volatility, trailing 12-month dividend yield; every
  factor is strictly lookahead-safe (only ever sees data before the date
  it's computed as of). Also owns `tickers_with_complete_history`, which
  `backtest.run_backtest` uses to automatically exclude any ticker too
  young for the backtest's full date range (an IPO or spinoff after it
  starts), rather than crashing on missing history
- **Ranking** (`src/models/ranker.py`, `src/models/lgbm_ranker.py`) — a
  composite z-score across the three factors (default), a pure-momentum
  alternative for comparison, and a LightGBM ranker that predicts next-month
  return and is refit fresh at every rebalance — all interchangeable via
  `backtest.run_backtest`'s `rank_fn` parameter
- **Optimisation** (`src/portfolio/optimise.py`) — mean-variance max Sharpe
  with Ledoit-Wolf covariance shrinkage (default), and a from-scratch Equal
  Risk Contribution risk-parity optimiser (convex reformulation via
  `cvxpy`) as an alternative via `optimiser_fn`; both bounded to avoid
  single-name concentration, with an automatic equal-weight fallback if
  the optimiser becomes numerically infeasible during extreme market
  stress (verified against the March 2020 COVID-crash rebalance)
- **Risk metrics** (`src/risk/metrics.py`) — Sharpe, Sortino, max
  drawdown, historical VaR, Monte Carlo VaR, expected shortfall (CVaR),
  and VaR breaches, at 95% and 99% confidence; Sharpe/Sortino accept
  either a flat risk-free rate or a real time-varying one (FRED)
- **Walk-forward backtest** (`src/portfolio/backtest.py`) — monthly
  rebalancing on a rolling 3-year training window from 2015-present,
  transaction costs computed from actual price-drifted turnover (not just
  target-to-target), compared against equal-weight and S&P 500 buy-and-hold
  benchmarks
- **Pipeline** (`src/data/pipeline.py`) — ingest → backtest → risk metrics
  in one call: the two data fetches run concurrently, each retrying
  automatically on failure and falling back to cached data if a live
  fetch can't be completed
- **Dashboard** (`app.py`) — Streamlit app: pick a ticker universe, run
  the backtest, browse holdings at any historical rebalance date, compare
  risk metrics, see the cumulative-growth chart
- **Notebooks** (`notebooks/`) — `01_pass1_exploration.ipynb` (the
  original stage-by-stage walkthrough), `02_sortino_optimisation_experiment.ipynb`
  (a standalone Sortino-vs-Sharpe comparison), `03_pass2_exploration.ipynb`
  (every Pass 2 addition demonstrated and compared against the Pass 1
  baseline, including the full S&P 500 universe run)

**What the default pipeline/dashboard actually run**, to be precise: `app.py`
and a bare `pipeline.run_pipeline()` call use the composite ranker, max-Sharpe
optimiser, the ~39-ticker dev universe, and a flat 2% risk-free rate — the
fast, simple configuration. LightGBM, risk-parity, the full S&P 500 universe,
and the real FRED rate are all built, tested, and demonstrated in
`03_pass2_exploration.ipynb`, but wiring them up as the pipeline/dashboard's
*default* is a deliberately separate, not-yet-done step (see Configuration).

**Explicitly out of scope**: a fundamentals-based quality factor (ROE,
leverage, earnings quality). Investigated — yfinance's fundamentals data
doesn't go back far enough for this project's backtest range; SEC EDGAR's
XBRL API would be the proper free source if ever revisited — but skipped as
not worth the lift for what it would add here.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running things

**Tests** — fast and fully offline (synthetic data, no network calls):

```bash
source .venv/bin/activate
python3 -m pytest tests/unit -v
```

**The notebooks** — pull real data via yfinance/FRED/Wikipedia on first
run, cached to `data/cache/` after that. `03_pass2_exploration.ipynb` is
the most complete walkthrough (LightGBM, risk-parity, the FRED rate, and
the full S&P 500 universe, each compared against the Pass 1 baseline):

```bash
source .venv/bin/activate
jupyter lab notebooks/03_pass2_exploration.ipynb
# or open it in VS Code and select the "portfolio-constructor" kernel
```

**The dashboard**:

```bash
source .venv/bin/activate
streamlit run app.py
```

**The pipeline directly**, e.g. from a Python shell (retries the data
fetches automatically, logs each step):

```python
from src.data import pipeline
results = pipeline.run_pipeline()
results["risk_comparison"]
```

**A non-default configuration** — e.g. LightGBM + risk-parity + the full
S&P 500, the way `03_pass2_exploration.ipynb` demonstrates — by calling
`backtest.run_backtest` directly rather than through `pipeline.run_pipeline`:

```python
from src.data import ingest
from src.models import lgbm_ranker
from src.portfolio import backtest, optimise
from src import config

tickers = ingest.load_or_fetch_sp500_tickers()
prices_long = ingest.load_or_fetch_prices(tickers + [config.BENCHMARK_TICKER], config.DATA_START_DATE, config.DATA_END_DATE)
dividends = ingest.load_or_fetch_dividends(tickers, config.DATA_START_DATE, config.DATA_END_DATE)
prices = ingest.to_wide_adj_close(prices_long)

daily_returns, rebalance_log = backtest.run_backtest(
    prices, dividends, tickers=tickers, rank_fn=lgbm_ranker.rank_by_lgbm, optimiser_fn=optimise.risk_parity_weights
)
```

## Configuration

Every constant — the ticker universe, date ranges, rebalance frequency,
transaction cost, factor lookback windows, optimiser bounds, risk-metric
settings — lives in `src/config.py`. Nothing is hardcoded elsewhere, so any
assumption can be found and changed in one place.

## Project structure

```
portfolio-constructor/
├── src/
│   ├── config.py            # every constant used across the pipeline
│   ├── data/
│   │   ├── ingest.py         # yfinance/FRED/Wikipedia fetching (no caching logic itself)
│   │   ├── storage.py        # DuckDB connection, schema, incremental caching
│   │   ├── features.py       # alpha factors + tickers_with_complete_history
│   │   └── pipeline.py       # ingest -> backtest -> risk metrics, concurrent fetches
│   ├── models/
│   │   ├── ranker.py         # composite z-score ranking (+ momentum-only variant)
│   │   └── lgbm_ranker.py     # LightGBM ranker, refit fresh at every rebalance
│   ├── portfolio/
│   │   ├── optimise.py       # max-Sharpe (Ledoit-Wolf) + risk-parity optimisers
│   │   └── backtest.py       # walk-forward backtest engine
│   └── risk/
│       └── metrics.py        # Sharpe, Sortino, drawdown, VaR, CVaR, breaches
├── app.py                    # Streamlit dashboard
├── notebooks/
│   ├── 01_pass1_exploration.ipynb
│   ├── 02_sortino_optimisation_experiment.ipynb
│   └── 03_pass2_exploration.ipynb
├── tests/unit/                # fast, offline, synthetic-data tests
└── data/cache/                # gitignored DuckDB cache (created on first run)
```
