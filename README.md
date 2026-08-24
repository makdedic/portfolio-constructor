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
placeholders, the full S&P 500 as an available universe, and proper
storage. Both passes are complete.

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

**What runs by default**, to be precise: `app.py` on first load, and a bare
`pipeline.run_pipeline()` call, both use the composite ranker, max-Sharpe
optimiser, the ~39-ticker dev universe, and a flat 2% risk-free rate — the
fast, simple configuration. Every other combination — LightGBM, risk-parity,
the full S&P 500 universe, the real FRED rate, and any mix of them — is a
sidebar control away in the dashboard, or an argument away in
`pipeline.run_pipeline(...)` (see Configuration).

**Explicitly out of scope**: a fundamentals-based quality factor (ROE,
leverage, earnings quality). Investigated — yfinance's fundamentals data
doesn't go back far enough for this project's backtest range; SEC EDGAR's
XBRL API would be the proper free source if ever revisited — but skipped as
not worth the lift for what it would add here.

## Methodology

The mathematics behind the three components most worth understanding line by line:

- **Mean-variance max-Sharpe** (`optimise.max_sharpe_weights`) finds the
  weights `w` that maximise `(w'μ − r_f) / sqrt(w'Σw)` — expected portfolio
  return in excess of the risk-free rate `r_f`, divided by portfolio
  volatility. `μ` is each stock's expected return (here, trailing historical
  average); `Σ` is the covariance matrix between every pair of stocks, so
  `w'Σw` (portfolio variance) captures diversification, not just each
  stock's own volatility. With only a few years of daily data, the raw
  sample covariance matrix has too many pairwise terms to estimate reliably
  and comes out noisy — Ledoit-Wolf shrinkage blends it with a simpler,
  more stable target matrix, trading a small amount of bias for a
  meaningful cut in estimation error. Weights are also capped at
  `MAX_WEIGHT_PER_STOCK` so the optimiser can't concentrate everything into
  one name.
- **Risk parity / Equal Risk Contribution** (`optimise.risk_parity_weights`)
  never looks at expected returns at all. A stock's contribution to total
  portfolio risk is `w_i × (Σw)_i` — its weight times how much it co-moves
  with the rest of the portfolio — and "equal risk contribution" means
  solving for weights where that quantity is the same for every holding, so
  no single stock dominates the portfolio's variance regardless of its raw
  weight. It's solved via the standard convex reformulation (Maillard,
  Roncalli & Teiletche, 2010) — minimising `0.5 w'Σw − Σ ln(w_i)` over
  `w > 0` — whose own optimum works out algebraically to that same
  equal-risk-contribution condition, so a convex, easy-to-solve problem
  stands in for the harder direct one.
- **VaR and CVaR** (`risk/metrics.py`) each answer "how bad could a day
  get," differently. Historical VaR at 95% confidence is just the 5th
  percentile of the *actual* historical daily returns — the loss level
  breached on 5% of real trading days, no assumption about the shape of the
  distribution. Monte Carlo VaR instead fits a normal distribution to the
  historical mean and standard deviation, draws 10,000 simulated daily
  returns from it, and reads the same percentile off the simulated data —
  smoother, but only as good as the assumption that returns are actually
  normally distributed (real markets have fatter tails than a normal curve
  predicts). CVaR (Expected Shortfall) goes one step further: instead of
  the threshold loss at that percentile, it's the *average* loss on every
  day that breaches it — "given a bad day happened, how bad was it on
  average," which matters more than the threshold alone in an actual tail
  event.

**What these actually produce**, whole-backtest Sharpe (2015-2026, net of
costs including the size-aware surcharge above, `03_pass2_exploration.ipynb`):

| Strategy | Sharpe |
|---|---|
| Equal-weight | 0.97 |
| Momentum-only | 0.91 |
| LightGBM | 0.73 |
| SPY (benchmark) | 0.72 |
| Risk-parity (composite ranker) | 0.70 |
| Composite (default, max-Sharpe) | 0.65 |

Worth stating plainly rather than only showcasing the flattering numbers:
equal-weight and momentum-only clearly beat SPY, and LightGBM edges it out
narrowly, but **the actual default configuration of this project — the
composite ranker with max-Sharpe optimisation — does not beat a
buy-and-hold SPY position** on a risk-adjusted basis, and neither does
risk-parity. This isn't simply "high-turnover strategies get punished
harder": risk-parity has the *lowest* turnover and lowest total cost of
the four active strategies here, yet still underperforms — its edge over
SPY was thin even before the cost model above was made more realistic.
None of this project's ranking or optimisation machinery reliably beats
"just hold everything at the same weight," at least not on this specific
universe and backtest window.

**Why**, broken out gross vs net of costs — it isn't one story:

| Strategy | Gross Sharpe | Net Sharpe | Cost drag |
|---|---|---|---|
| SPY | 0.72 | 0.72 | 0.00 |
| Composite (default) | 0.72 | 0.65 | 0.07 |
| Risk-parity | 0.74 | 0.71 | 0.04 |
| LightGBM | 0.83 | 0.73 | 0.10 |
| Momentum-only | 0.96 | 0.91 | 0.05 |

Composite and risk-parity barely beat SPY *even before costs* — their
gross Sharpe is essentially tied with SPY's, not genuinely ahead, so this
isn't a costs problem at all: the ranking and optimisation aren't adding
real value here in the first place. LightGBM is the opposite: its gross
Sharpe clearly beats SPY, so the stock-picking genuinely works — but it
also has by far the highest turnover of any strategy here, so it pays the
largest cost drag and ends up with only a bare net edge. Momentum-only is
the one variant where a real edge survives intact: clearly ahead gross,
and still comfortably ahead net.

## Limitations

Where these results should and shouldn't be trusted:

- **Survivorship bias** — the S&P 500 universe uses today's constituent
  list applied retroactively across the whole 2015-2026 backtest.
  Companies removed from the index over that period (usually because they
  performed badly) never appear as candidates, which flatters the
  full-universe results more than the headline numbers alone suggest.
- **One historical path** — the backtest runs once over a single realised
  sequence of market history, not resampled or stress-tested against
  alternate scenarios. Split into three regimes to check this directly
  (`03_pass2_exploration.ipynb`, section 11): the default strategy's Sharpe
  is *below* SPY's in all three — 0.68 vs 0.79 in the calm 2015-2019 bull
  market, 0.81 vs 0.88 through the COVID crash and recovery, 0.59 vs 0.62
  through the 2022 rate-hike bear market and recovery. That's a real
  reversal from an earlier version of this check, which found the
  opposite conclusion — the difference is the transaction cost model
  becoming more realistic (see Methodology), not a change to the
  underlying strategy. The default configuration's edge over SPY was thin
  enough that a fuller accounting of trading costs erases it entirely,
  consistently across all three regimes rather than just one. A useful
  reminder of how sensitive a backtest's conclusion can be to a single
  modelling assumption, and why this section exists.
- **Weights held fixed between rebalances** — positions drift with price
  and aren't corrected until the next monthly rebalance rather than
  managed continuously. This is a chosen simplification and is a
  gap from how the strategy would actually trade.
- **Simplified transaction costs** — a flat 10bps per rebalance regardless
  of trade size or liquidity, with no market impact or bid-ask spread
  modelling. Real execution costs for a less liquid holding would be
  worse than this accounts for.
- **Small per-fit training data for LightGBM** — each monthly refit sees
  roughly 1,000-1,300 rows (~39 tickers × ~30 monthly snapshots).
  Hyperparameters are kept conservative specifically because of this, but
  a model refit on that little data every month is still genuinely prone
  to noisy, unstable predictions.
- **Iterative development risk** — the backtest start date was adjusted
  partway through development after checking which tickers a given date
  would include, not fixed beforehand. Several rankers and
  optimisers were also compared on the same backtest window, and the
  best-performing combination (LightGBM on the full S&P 500) is the one
  most prominently reported.

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
S&P 500 + the real FRED rate, the way `03_pass2_exploration.ipynb`
demonstrates — by passing different arguments to `pipeline.run_pipeline`:

```python
from src.data import ingest, pipeline
from src.models import lgbm_ranker
from src.portfolio import optimise

tickers = ingest.load_or_fetch_sp500_tickers()
results = pipeline.run_pipeline(
    tickers=tickers,
    rank_fn=lgbm_ranker.rank_by_lgbm,
    optimiser_fn=optimise.risk_parity_weights,
    use_real_risk_free_rate=True,
)
```

## Deployment

The dashboard runs on Streamlit Community Cloud. Getting it to work
reliably there surfaced real production issues that a local dev loop never
would have — worth documenting honestly rather than glossing over:

- **Yahoo rate-limits Streamlit Cloud's shared IP, independent of request
  volume.** Confirmed directly: even the small ~39-ticker default failed
  with `YFRateLimitError` after exhausting retries, on an IP this app's own
  traffic alone wouldn't come close to tripping. Two mitigations, neither a
  full guarantee against a determined block: prices and dividends both
  fetch through one batched, retried `yf.download()` call
  (`ingest._download_with_retry`) instead of hundreds of fragile
  per-ticker requests, and that response is validated before it's trusted
  — `yf.download()` can silently return an all-NaN column for a ticker
  with no exception raised at all (confirmed directly against a real
  corrupted fetch), so a wholly-empty ticker triggers a retry instead of
  getting cached as if it were real data.
- **The deployed app ships with a pre-fetched data snapshot** (`data/seed/`,
  built by `scripts/build_seed.py`) so it starts already warm instead of
  needing to fetch anything live on a fresh deploy. If a live "top-up"
  fetch to reach today's date fails, `ingest.load_or_fetch_prices` /
  `load_or_fetch_dividends` fall back to whatever's already cached instead
  of crashing — the app shows real data, just not current as of today.
  Trade-off: the deployed data is only as fresh as the last time the seed
  was rebuilt and redeployed, not automatically kept current. To refresh
  it:

  ```bash
  python scripts/build_seed.py
  git add data/seed/
  git commit -m "chore: refresh seed data"
  git push
  ```

- **DuckDB connections are always explicitly closed** (`with
  storage.get_connection() as conn:` in every `ingest.py` caller).
  Streamlit keeps one Python process running across every user
  interaction, unlike a short-lived script — a connection left open by an
  interrupted run doesn't get cleaned up by the process exiting, and can
  outlive it long enough to collide with the next run's own write. DuckDB
  allows only one writer per file; confirmed directly in production as a
  `TransactionException` write-write conflict before this fix.
- **No orchestration framework runs in the deployed path.** This was
  originally a Prefect flow; Prefect's own local orchestration server
  turned out to be the actual source of the worst deployment failures — an
  ephemeral-server startup timeout, a port collision with a botched
  shutdown, and a sluggish, unresponsive UI while it managed its own async
  lifecycle inside Streamlit's long-running process — all for
  retry/concurrency behaviour `ingest.py` already provides on its own.
  `pipeline.py` now calls the same ingest/backtest/metrics functions
  directly, with a plain `ThreadPoolExecutor` for the two concurrent
  fetches.
- **Tests run automatically on every push and pull request**
  (`.github/workflows/tests.yml`), against a clean install from
  `requirements.txt` — verified directly in a fresh virtualenv, not just
  assumed to work.

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
├── scripts/
│   └── build_seed.py         # rebuilds data/seed/ - run whenever deployed data should refresh
├── notebooks/
│   ├── 01_pass1_exploration.ipynb
│   ├── 02_sortino_optimisation_experiment.ipynb
│   └── 03_pass2_exploration.ipynb
├── tests/unit/                # fast, offline, synthetic-data tests
├── .github/workflows/tests.yml  # runs tests/unit on every push and pull request
└── data/
    ├── seed/                  # committed S&P 500 snapshot - ships with the deployed app
    └── cache/                 # gitignored local DuckDB cache (created on first run)
```
