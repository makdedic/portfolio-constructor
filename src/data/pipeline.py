"""Runs the full pipeline end to end: ingest data, walk-forward backtest,
risk metrics.

A plain function for now — no Prefect orchestration yet. Wrapping this in
@flow/@task later is additive (decorators around the same logic), not a
rewrite; introducing Prefect now would mean learning it at the same time as
everything else here, which works against keeping things simple.
"""

import pandas as pd

from src import config
from src.data import ingest
from src.portfolio import backtest
from src.risk import metrics


def run_pipeline(
    tickers: list[str] = config.TICKER_UNIVERSE,
    benchmark_ticker: str = config.BENCHMARK_TICKER,
    start: pd.Timestamp = config.BACKTEST_START_DATE,
    end: pd.Timestamp = config.BACKTEST_END_DATE,
) -> dict:
    """Ingest, run the walk-forward backtest, then compute risk metrics.

    Returns a dict with the ingested prices/dividends, the backtest's
    daily_returns and rebalance_log, and a risk-metrics comparison table
    (strategy vs. equal-weight vs. benchmark, all net of transaction costs)
    — everything app.py needs to render, in one call.
    """
    all_tickers = tickers + [benchmark_ticker]
    prices_long = ingest.load_or_fetch_prices(all_tickers, config.DATA_START_DATE, config.DATA_END_DATE)
    dividends = ingest.load_or_fetch_dividends(tickers, config.DATA_START_DATE, config.DATA_END_DATE)
    prices = ingest.to_wide_adj_close(prices_long)

    daily_returns, rebalance_log = backtest.run_backtest(
        prices, dividends, tickers=tickers, benchmark_ticker=benchmark_ticker, start=start, end=end
    )

    risk_comparison = pd.DataFrame(
        {
            "strategy": metrics.compute_all(daily_returns["strategy_net"]),
            "equal_weight": metrics.compute_all(daily_returns["equal_weight_net"]),
            benchmark_ticker: metrics.compute_all(daily_returns["sp500_net"]),
        }
    )

    return {
        "prices": prices,
        "dividends": dividends,
        "daily_returns": daily_returns,
        "rebalance_log": rebalance_log,
        "risk_comparison": risk_comparison,
    }
