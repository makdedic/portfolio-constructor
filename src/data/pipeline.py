"""Runs the full pipeline end to end: ingest data, walk-forward backtest,
risk metrics — orchestrated with Prefect.

ingest.py/backtest.py/metrics.py stay plain, Prefect-agnostic functions
(they're unit-tested and used directly in notebooks with no orchestration
context) — the @task/@flow decoration lives only here, as thin wrappers
around calls to that existing logic.
"""

from datetime import date

import pandas as pd
from prefect import flow, get_run_logger, task

from src import config
from src.data import ingest
from src.models import ranker
from src.portfolio import backtest, optimise
from src.risk import metrics


@task(retries=config.PREFECT_TASK_RETRIES, retry_delay_seconds=config.PREFECT_RETRY_DELAY_SECONDS)
def fetch_prices_task(tickers: list[str], start, end) -> pd.DataFrame:
    """Retried: the most network-dependent, failure-prone step (yfinance)."""
    prices = ingest.load_or_fetch_prices(tickers, start, end)
    get_run_logger().info(f"fetched prices for {len(tickers)} tickers ({len(prices)} rows)")
    return prices


@task(retries=config.PREFECT_TASK_RETRIES, retry_delay_seconds=config.PREFECT_RETRY_DELAY_SECONDS)
def fetch_dividends_task(tickers: list[str], start, end) -> pd.DataFrame:
    """Retried, same reason as fetch_prices_task."""
    dividends = ingest.load_or_fetch_dividends(tickers, start, end)
    get_run_logger().info(f"fetched dividends for {len(tickers)} tickers ({len(dividends)} payments)")
    return dividends


@task(retries=config.PREFECT_TASK_RETRIES, retry_delay_seconds=config.PREFECT_RETRY_DELAY_SECONDS)
def fetch_risk_free_rate_task(start, end) -> pd.Series:
    """Retried, same reason as the price/dividend fetches - FRED has shown
    real flakiness this session (request timeouts).
    """
    rate_df = ingest.load_or_fetch_risk_free_rate(start, end)
    rate = rate_df.set_index("date")["risk_free_rate_annual"]
    get_run_logger().info(f"fetched risk-free rate: {len(rate)} days")
    return rate


@task
def run_backtest_task(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    tickers: list[str],
    benchmark_ticker: str,
    start: date,
    end: date,
    rank_fn,
    optimiser_fn,
    risk_free_rate=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """No retries — a deterministic computation on already-fetched data, not
    a flaky call retrying would help. Wrapped for observability: it shows up
    as its own step in Prefect's run history.
    """
    daily_returns, rebalance_log = backtest.run_backtest(
        prices,
        dividends,
        tickers=tickers,
        benchmark_ticker=benchmark_ticker,
        start=start,
        end=end,
        rank_fn=rank_fn,
        optimiser_fn=optimiser_fn,
        risk_free_rate=risk_free_rate,
    )
    get_run_logger().info(f"backtest complete: {len(rebalance_log)} rebalances, {start} to {end}")
    return daily_returns, rebalance_log


@task
def compute_risk_metrics_task(daily_returns: pd.DataFrame, benchmark_ticker: str, risk_free_rate=None) -> pd.DataFrame:
    """No retries, same reason as run_backtest_task."""
    # metrics.compute_all defaults risk_free_rate_annual to the flat config
    # constant on its own - only override that default when a real series
    # was actually fetched.
    rate_kwargs = {} if risk_free_rate is None else {"risk_free_rate_annual": risk_free_rate}
    risk_comparison = pd.DataFrame(
        {
            "strategy": metrics.compute_all(daily_returns["strategy_net"], **rate_kwargs),
            "equal_weight": metrics.compute_all(daily_returns["equal_weight_net"], **rate_kwargs),
            benchmark_ticker: metrics.compute_all(daily_returns["sp500_net"], **rate_kwargs),
        }
    )
    get_run_logger().info(f"strategy Sharpe: {risk_comparison.loc['sharpe_ratio', 'strategy']:.3f}")
    return risk_comparison


@flow(name="portfolio-pipeline")
def run_pipeline(
    tickers: list[str] = config.TICKER_UNIVERSE,
    benchmark_ticker: str = config.BENCHMARK_TICKER,
    start: date = config.BACKTEST_START_DATE,
    end: date = config.BACKTEST_END_DATE,
    rank_fn=ranker.rank_by_composite_score,
    optimiser_fn=optimise.max_sharpe_weights,
    use_real_risk_free_rate: bool = False,
) -> dict:
    """Ingest, run the walk-forward backtest, then compute risk metrics.

    rank_fn/optimiser_fn select the ranking rule and portfolio construction
    method, same as backtest.run_backtest's own parameters of the same name
    (e.g. pass ranker.rank_by_momentum_only or lgbm_ranker.rank_by_lgbm;
    optimise.max_sharpe_weights or optimise.risk_parity_weights).

    use_real_risk_free_rate fetches the actual daily FRED 3-month T-bill
    history and uses it in place of the flat config.RISK_FREE_RATE_ANNUAL
    constant, for both the max-Sharpe hurdle and the reported Sharpe/Sortino.
    False by default so the fast path stays fast and network-free for this
    piece - no FRED call happens unless asked for.

    Returns a dict with the ingested prices/dividends, the backtest's
    daily_returns and rebalance_log, and a risk-metrics comparison table
    (strategy vs. equal-weight vs. benchmark, all net of transaction costs)
    — everything app.py needs to render, in one call.

    The fetches are independent of each other, so they're submitted to run
    concurrently rather than one after another. Different ticker sets (e.g.
    the dev universe vs. the full S&P 500) share cached data via
    src.data.storage rather than needing separate caches.
    """
    all_tickers = tickers + [benchmark_ticker]
    prices_future = fetch_prices_task.submit(all_tickers, config.DATA_START_DATE, config.DATA_END_DATE)
    dividends_future = fetch_dividends_task.submit(tickers, config.DATA_START_DATE, config.DATA_END_DATE)
    risk_free_rate_future = None
    if use_real_risk_free_rate:
        risk_free_rate_future = fetch_risk_free_rate_task.submit(config.DATA_START_DATE, config.DATA_END_DATE)

    prices_long = prices_future.result()
    dividends = dividends_future.result()
    risk_free_rate = risk_free_rate_future.result() if risk_free_rate_future is not None else None
    prices = ingest.to_wide_adj_close(prices_long)

    daily_returns, rebalance_log = run_backtest_task(
        prices, dividends, tickers, benchmark_ticker, start, end, rank_fn, optimiser_fn, risk_free_rate
    )
    risk_comparison = compute_risk_metrics_task(daily_returns, benchmark_ticker, risk_free_rate)

    return {
        "prices": prices,
        "dividends": dividends,
        "daily_returns": daily_returns,
        "rebalance_log": rebalance_log,
        "risk_comparison": risk_comparison,
    }
