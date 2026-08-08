"""Runs the full pipeline end to end: ingest data, walk-forward backtest,
risk metrics.

ingest.py/backtest.py/metrics.py stay plain functions with no orchestration
awareness (they're unit-tested and used directly in notebooks too) - this
module just calls them in the right order. The two independent fetches
(prices, dividends, plus the risk-free rate when asked for) run concurrently
via a plain ThreadPoolExecutor rather than a dedicated orchestration
framework - this used to be a Prefect flow, but Prefect's own local
orchestration server proved to be the single biggest source of deployment
failures (startup timeouts, port collisions, a stale/unresponsive UI while
it juggled its own async lifecycle), all for retry/concurrency behaviour
ingest.py already provides on its own (_download_with_retry, and falling
back to cached data on failure) - infrastructure the app didn't need.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd

from src import config
from src.data import ingest
from src.models import ranker
from src.portfolio import backtest, optimise
from src.risk import metrics

logger = logging.getLogger(__name__)


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

    The fetches are independent of each other, so they run concurrently
    rather than one after another. Different ticker sets (e.g. the dev
    universe vs. the full S&P 500) share cached data via src.data.storage
    rather than needing separate caches.
    """
    all_tickers = tickers + [benchmark_ticker]

    with ThreadPoolExecutor(max_workers=3) as executor:
        prices_future = executor.submit(
            ingest.load_or_fetch_prices, all_tickers, config.DATA_START_DATE, config.DATA_END_DATE
        )
        dividends_future = executor.submit(
            ingest.load_or_fetch_dividends, tickers, config.DATA_START_DATE, config.DATA_END_DATE
        )
        risk_free_rate_future = None
        if use_real_risk_free_rate:
            risk_free_rate_future = executor.submit(
                ingest.load_or_fetch_risk_free_rate, config.DATA_START_DATE, config.DATA_END_DATE
            )

        prices_long = prices_future.result()
        dividends = dividends_future.result()
        risk_free_rate = None
        if risk_free_rate_future is not None:
            rate_df = risk_free_rate_future.result()
            risk_free_rate = rate_df.set_index("date")["risk_free_rate_annual"]

    logger.info("fetched prices for %d tickers (%d rows)", len(all_tickers), len(prices_long))
    logger.info("fetched dividends for %d tickers (%d payments)", len(tickers), len(dividends))
    if risk_free_rate is not None:
        logger.info("fetched risk-free rate (%d days)", len(risk_free_rate))
    prices = ingest.to_wide_adj_close(prices_long)

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
    logger.info("backtest complete: %d rebalances, %s to %s", len(rebalance_log), start, end)

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
    logger.info("strategy Sharpe: %.3f", risk_comparison.loc["sharpe_ratio", "strategy"])

    return {
        "prices": prices,
        "dividends": dividends,
        "daily_returns": daily_returns,
        "rebalance_log": rebalance_log,
        "risk_comparison": risk_comparison,
    }
