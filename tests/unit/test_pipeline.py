"""Guards the pipeline's orchestration/composition, not the underlying
logic - ingest/backtest/metrics already have their own tests (39 of them).
This checks that run_pipeline wires those pieces together correctly: right
arguments reaching the right calls, cache_name actually threading through,
and retries configured on the tasks that need them.
"""

from unittest.mock import patch

import pandas as pd

from src import config
from src.data import pipeline


def _fake_daily_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy_net": [0.01, -0.01],
            "equal_weight_net": [0.005, -0.005],
            "sp500_net": [0.008, -0.002],
        }
    )


def test_run_pipeline_threads_cache_names_to_the_right_fetch_calls():
    with (
        patch("src.data.pipeline.ingest.load_or_fetch_prices") as fetch_prices_mock,
        patch("src.data.pipeline.ingest.load_or_fetch_dividends") as fetch_dividends_mock,
        patch("src.data.pipeline.ingest.to_wide_adj_close") as to_wide_mock,
        patch("src.data.pipeline.backtest.run_backtest") as run_backtest_mock,
        patch("src.data.pipeline.metrics.compute_all") as compute_all_mock,
    ):
        fetch_prices_mock.return_value = pd.DataFrame({"date": [], "ticker": [], "adj_close": []})
        fetch_dividends_mock.return_value = pd.DataFrame()
        to_wide_mock.return_value = pd.DataFrame()
        run_backtest_mock.return_value = (_fake_daily_returns(), pd.DataFrame())
        compute_all_mock.return_value = {"sharpe_ratio": 1.0}

        pipeline.run_pipeline(
            tickers=["AAPL", "MSFT"],
            benchmark_ticker="SPY",
            prices_cache_name="prices_test",
            dividends_cache_name="dividends_test",
        )

        prices_call = fetch_prices_mock.call_args
        assert prices_call.kwargs["cache_name"] == "prices_test"
        assert set(prices_call.args[0]) == {"AAPL", "MSFT", "SPY"}  # benchmark included in the price fetch

        dividends_call = fetch_dividends_mock.call_args
        assert dividends_call.kwargs["cache_name"] == "dividends_test"
        assert dividends_call.args[0] == ["AAPL", "MSFT"]  # benchmark excluded from the dividend fetch


def test_run_pipeline_calls_backtest_with_the_requested_tickers_and_returns_expected_keys():
    with (
        patch("src.data.pipeline.ingest.load_or_fetch_prices") as fetch_prices_mock,
        patch("src.data.pipeline.ingest.load_or_fetch_dividends") as fetch_dividends_mock,
        patch("src.data.pipeline.ingest.to_wide_adj_close") as to_wide_mock,
        patch("src.data.pipeline.backtest.run_backtest") as run_backtest_mock,
        patch("src.data.pipeline.metrics.compute_all") as compute_all_mock,
    ):
        fetch_prices_mock.return_value = pd.DataFrame({"date": [], "ticker": [], "adj_close": []})
        fetch_dividends_mock.return_value = pd.DataFrame()
        to_wide_mock.return_value = pd.DataFrame({"AAPL": [1.0], "MSFT": [2.0], "SPY": [3.0]})
        run_backtest_mock.return_value = (_fake_daily_returns(), pd.DataFrame({"turnover": [0.1]}))
        compute_all_mock.return_value = {"sharpe_ratio": 1.0}

        results = pipeline.run_pipeline(tickers=["AAPL", "MSFT"], benchmark_ticker="SPY")

        assert set(results.keys()) == {
            "prices",
            "dividends",
            "daily_returns",
            "rebalance_log",
            "risk_comparison",
        }

        run_backtest_mock.assert_called_once()
        _, backtest_kwargs = run_backtest_mock.call_args
        assert backtest_kwargs["tickers"] == ["AAPL", "MSFT"]
        assert backtest_kwargs["benchmark_ticker"] == "SPY"

        # One compute_all call each for strategy, equal_weight, and the benchmark.
        assert compute_all_mock.call_count == 3


def test_fetch_tasks_are_configured_with_retries_but_computation_tasks_are_not():
    assert pipeline.fetch_prices_task.retries == config.PREFECT_TASK_RETRIES
    assert pipeline.fetch_prices_task.retry_delay_seconds == config.PREFECT_RETRY_DELAY_SECONDS
    assert pipeline.fetch_dividends_task.retries == config.PREFECT_TASK_RETRIES
    assert pipeline.fetch_dividends_task.retry_delay_seconds == config.PREFECT_RETRY_DELAY_SECONDS

    # Deterministic computation on already-fetched data - retrying wouldn't
    # fix a failure here, so these stay unretried (wrapped only for
    # observability in Prefect's run history).
    assert pipeline.run_backtest_task.retries == 0
    assert pipeline.compute_risk_metrics_task.retries == 0
