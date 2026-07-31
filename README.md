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

## What's implemented (Pass 1 — thin vertical slice)

Every pipeline stage is connected end-to-end, using a ~39-ticker dev
universe rather than the full S&P 500, so the whole system is fast and
reliable to run while it's still being built out:

- **Ingestion** (`src/data/ingest.py`) — daily prices + dividends via
  yfinance, cached locally in DuckDB
- **Alpha factors** (`src/data/features.py`) — 12-1 month momentum,
  trailing 12-month volatility, trailing 12-month dividend yield; every
  factor is strictly lookahead-safe (only ever sees data before the date
  it's computed as of)
- **Ranking** (`src/models/ranker.py`) — a composite z-score across the
  three factors (`score_stocks`), plus a pure-momentum alternative
  (`score_by_momentum_only`) for comparison
- **Optimisation** (`src/portfolio/optimise.py`) — mean-variance max
  Sharpe with Ledoit-Wolf covariance shrinkage (PyPortfolioOpt), bounded
  to avoid single-name concentration
- **Risk metrics** (`src/risk/metrics.py`) — Sharpe, Sortino, max
  drawdown, historical VaR, Monte Carlo VaR, expected shortfall (CVaR),
  and VaR breaches, at 95% and 99% confidence
- **Walk-forward backtest** (`src/portfolio/backtest.py`) — monthly
  rebalancing on a rolling 3-year training window, transaction costs
  computed from actual price-drifted turnover (not just target-to-target),
  compared against equal-weight and S&P 500 buy-and-hold benchmarks
- **Pipeline** (`src/data/pipeline.py`) — a single `run_pipeline()` call
  chaining ingestion → backtest → risk metrics
- **Dashboard** (`app.py`) — Streamlit app: pick a ticker universe, run
  the backtest, browse holdings at any historical rebalance date, compare
  risk metrics, see the cumulative-growth chart
- **Notebook** (`notebooks/01_pass1_exploration.ipynb`) — scratch
  space for exploring the pipeline interactively, one stage at a time

**Deferred to a later pass** (breadth, not correctness — the six rules
above already hold in full in this pass): full S&P 500 universe, quality
factor, LightGBM ranker, risk-parity optimisation, Prefect orchestration,
DuckDB storage, FRED risk-free rate, full 2010-present backtest range,
dashboard polish.

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

**The notebook** — pulls real data via yfinance on first run, cached to
`data/cache/` after that:

```bash
source .venv/bin/activate
jupyter lab notebooks/01_pass1_exploration.ipynb
# or open it in VS Code and select the "portfolio-constructor" kernel
```

**The dashboard**:

```bash
source .venv/bin/activate
streamlit run app.py
```

**The pipeline directly**, e.g. from a Python shell:

```python
from src.data import pipeline
results = pipeline.run_pipeline()
results["risk_comparison"]
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
│   ├── config.py           # every constant used across the pipeline
│   ├── data/
│   │   ├── ingest.py        # yfinance price + dividend ingestion, DuckDB cache
│   │   ├── features.py      # alpha factors (momentum, low-vol, dividend yield)
│   │   └── pipeline.py      # ingest -> backtest -> risk metrics, one call
│   ├── models/
│   │   └── ranker.py        # composite z-score ranking (+ momentum-only variant)
│   ├── portfolio/
│   │   ├── optimise.py      # mean-variance max Sharpe, Ledoit-Wolf shrinkage
│   │   └── backtest.py      # walk-forward backtest engine
│   └── risk/
│       └── metrics.py       # Sharpe, Sortino, drawdown, VaR, CVaR, breaches
├── app.py                   # Streamlit dashboard
├── notebooks/
│   ├── 01_pass1_exploration.ipynb
│   ├── 02_sortino_optimisation_experiment.ipynb
│   └── 03_pass2_exploration.ipynb
├── tests/unit/               # fast, offline, synthetic-data tests
└── data/cache/               # gitignored DuckDB cache (created on first run)
```
