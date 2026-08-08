"""Guards the pipeline's composition, not the underlying logic -
ingest/backtest/metrics already have their own tests. This checks that
run_pipeline wires those pieces together correctly: right arguments
reaching the right calls.
"""

from unittest.mock import patch

import pandas as pd

from src.data import pipeline


def _fake_daily_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy_net": [0.01, -0.01],
            "equal_weight_net": [0.005, -0.005],
            "sp500_net": [0.008, -0.002],
        }
    )


def test_run_pipeline_routes_the_right_tickers_to_the_right_fetch_calls():
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

        pipeline.run_pipeline(tickers=["AAPL", "MSFT"], benchmark_ticker="SPY")

        prices_call = fetch_prices_mock.call_args
        assert set(prices_call.args[0]) == {"AAPL", "MSFT", "SPY"}  # benchmark included in the price fetch

        dividends_call = fetch_dividends_mock.call_args
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


def test_run_pipeline_passes_rank_fn_and_optimiser_fn_to_run_backtest():
    def custom_rank_fn(*args, **kwargs):
        pass

    def custom_optimiser_fn(*args, **kwargs):
        pass

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
            tickers=["AAPL", "MSFT"], rank_fn=custom_rank_fn, optimiser_fn=custom_optimiser_fn
        )

        _, backtest_kwargs = run_backtest_mock.call_args
        assert backtest_kwargs["rank_fn"] is custom_rank_fn
        assert backtest_kwargs["optimiser_fn"] is custom_optimiser_fn


def test_use_real_risk_free_rate_fetches_and_threads_it_through():
    fake_rate = pd.Series([0.02, 0.03], index=pd.bdate_range("2020-01-01", periods=2))

    with (
        patch("src.data.pipeline.ingest.load_or_fetch_prices") as fetch_prices_mock,
        patch("src.data.pipeline.ingest.load_or_fetch_dividends") as fetch_dividends_mock,
        patch("src.data.pipeline.ingest.load_or_fetch_risk_free_rate") as fetch_rate_mock,
        patch("src.data.pipeline.ingest.to_wide_adj_close") as to_wide_mock,
        patch("src.data.pipeline.backtest.run_backtest") as run_backtest_mock,
        patch("src.data.pipeline.metrics.compute_all") as compute_all_mock,
    ):
        fetch_prices_mock.return_value = pd.DataFrame({"date": [], "ticker": [], "adj_close": []})
        fetch_dividends_mock.return_value = pd.DataFrame()
        fetch_rate_mock.return_value = pd.DataFrame(
            {"date": fake_rate.index, "risk_free_rate_annual": fake_rate.values}
        )
        to_wide_mock.return_value = pd.DataFrame()
        run_backtest_mock.return_value = (_fake_daily_returns(), pd.DataFrame())
        compute_all_mock.return_value = {"sharpe_ratio": 1.0}

        pipeline.run_pipeline(tickers=["AAPL", "MSFT"], use_real_risk_free_rate=True)

        fetch_rate_mock.assert_called_once()
        # check_names/check_freq=False: fetch_risk_free_rate_task names the
        # series and its index loses the bdate_range freq metadata through
        # set_index - neither is the property this test cares about.
        _, backtest_kwargs = run_backtest_mock.call_args
        pd.testing.assert_series_equal(
            backtest_kwargs["risk_free_rate"], fake_rate, check_names=False, check_freq=False
        )
        _, compute_all_kwargs = compute_all_mock.call_args
        pd.testing.assert_series_equal(
            compute_all_kwargs["risk_free_rate_annual"], fake_rate, check_names=False, check_freq=False
        )


def test_default_run_pipeline_never_fetches_the_risk_free_rate():
    with (
        patch("src.data.pipeline.ingest.load_or_fetch_prices") as fetch_prices_mock,
        patch("src.data.pipeline.ingest.load_or_fetch_dividends") as fetch_dividends_mock,
        patch("src.data.pipeline.ingest.load_or_fetch_risk_free_rate") as fetch_rate_mock,
        patch("src.data.pipeline.ingest.to_wide_adj_close") as to_wide_mock,
        patch("src.data.pipeline.backtest.run_backtest") as run_backtest_mock,
        patch("src.data.pipeline.metrics.compute_all") as compute_all_mock,
    ):
        fetch_prices_mock.return_value = pd.DataFrame({"date": [], "ticker": [], "adj_close": []})
        fetch_dividends_mock.return_value = pd.DataFrame()
        to_wide_mock.return_value = pd.DataFrame()
        run_backtest_mock.return_value = (_fake_daily_returns(), pd.DataFrame())
        compute_all_mock.return_value = {"sharpe_ratio": 1.0}

        pipeline.run_pipeline(tickers=["AAPL", "MSFT"])

        fetch_rate_mock.assert_not_called()
        _, backtest_kwargs = run_backtest_mock.call_args
        assert backtest_kwargs["risk_free_rate"] is None
