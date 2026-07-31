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
from src.portfolio import backtest
from src.risk import metrics


@task(retries=config.PREFECT_TASK_RETRIES, retry_delay_seconds=config.PREFECT_RETRY_DELAY_SECONDS)
def fetch_prices_task(tickers: list[str], start, end, cache_name: str) -> pd.DataFrame:
    """Retried: the most network-dependent, failure-prone step (yfinance)."""
    prices = ingest.load_or_fetch_prices(tickers, start, end, cache_name=cache_name)
    get_run_logger().info(f"fetched prices for {len(tickers)} tickers ({len(prices)} rows)")
    return prices


@task(retries=config.PREFECT_TASK_RETRIES, retry_delay_seconds=config.PREFECT_RETRY_DELAY_SECONDS)
def fetch_dividends_task(tickers: list[str], start, end, cache_name: str) -> pd.DataFrame:
    """Retried, same reason as fetch_prices_task."""
    dividends = ingest.load_or_fetch_dividends(tickers, start, end, cache_name=cache_name)
    get_run_logger().info(f"fetched dividends for {len(tickers)} tickers ({len(dividends)} payments)")
    return dividends


@task
def run_backtest_task(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    tickers: list[str],
    benchmark_ticker: str,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """No retries — a deterministic computation on already-fetched data, not
    a flaky call retrying would help. Wrapped for observability: it shows up
    as its own step in Prefect's run history.
    """
    daily_returns, rebalance_log = backtest.run_backtest(
        prices, dividends, tickers=tickers, benchmark_ticker=benchmark_ticker, start=start, end=end
    )
    get_run_logger().info(f"backtest complete: {len(rebalance_log)} rebalances, {start} to {end}")
    return daily_returns, rebalance_log


@task
def compute_risk_metrics_task(daily_returns: pd.DataFrame, benchmark_ticker: str) -> pd.DataFrame:
    """No retries, same reason as run_backtest_task."""
    risk_comparison = pd.DataFrame(
        {
            "strategy": metrics.compute_all(daily_returns["strategy_net"]),
            "equal_weight": metrics.compute_all(daily_returns["equal_weight_net"]),
            benchmark_ticker: metrics.compute_all(daily_returns["sp500_net"]),
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
    prices_cache_name: str = "prices",
    dividends_cache_name: str = "dividends",
) -> dict:
    """Ingest, run the walk-forward backtest, then compute risk metrics.

    prices_cache_name/dividends_cache_name distinguish runs against
    different universes (e.g. the dev ticker subset vs. the full S&P 500,
    see ingest.load_or_fetch_prices) so they don't clobber or misread each
    other's cache file — pass distinct names whenever tickers isn't
    config.TICKER_UNIVERSE.

    Returns a dict with the ingested prices/dividends, the backtest's
    daily_returns and rebalance_log, and a risk-metrics comparison table
    (strategy vs. equal-weight vs. benchmark, all net of transaction costs)
    — everything app.py needs to render, in one call.

    The two fetches are independent of each other, so they're submitted to
    run concurrently rather than one after another.
    """
    all_tickers = tickers + [benchmark_ticker]
    prices_future = fetch_prices_task.submit(
        all_tickers, config.DATA_START_DATE, config.DATA_END_DATE, prices_cache_name
    )
    dividends_future = fetch_dividends_task.submit(
        tickers, config.DATA_START_DATE, config.DATA_END_DATE, dividends_cache_name
    )

    prices_long = prices_future.result()
    dividends = dividends_future.result()
    prices = ingest.to_wide_adj_close(prices_long)

    daily_returns, rebalance_log = run_backtest_task(prices, dividends, tickers, benchmark_ticker, start, end)
    risk_comparison = compute_risk_metrics_task(daily_returns, benchmark_ticker)

    return {
        "prices": prices,
        "dividends": dividends,
        "daily_returns": daily_returns,
        "rebalance_log": rebalance_log,
        "risk_comparison": risk_comparison,
    }
