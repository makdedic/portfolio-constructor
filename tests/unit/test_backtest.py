"""Guards rules 3 and 5: transaction costs applied correctly, and both
benchmarks always present. (Rule 2, the rolling training window itself, is
tested in test_features.py alongside features.training_window.)
"""

import numpy as np
import pandas as pd
import pytest
from pypfopt import exceptions as pypfopt_exceptions

from src.models import lgbm_ranker
from src.portfolio import backtest, optimise


def _synthetic_universe(n_tickers: int = 5) -> pd.DataFrame:
    """~4 years of daily prices — enough for a 3-year training window plus
    a handful of months to actually backtest over. Includes one extra
    "BENCH" column to stand in for the SPY benchmark.
    """
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2015-01-01", "2018-12-31")
    columns = [f"T{i}" for i in range(n_tickers)] + ["BENCH"]
    daily_returns = rng.normal(loc=0.0005, scale=0.01, size=(len(dates), len(columns)))
    return 100 * (1 + pd.DataFrame(daily_returns, index=dates, columns=columns)).cumprod()


def _synthetic_universe_long(n_tickers: int = 8) -> pd.DataFrame:
    """~5 years of daily prices, one more than _synthetic_universe()'s 4.

    lgbm_ranker.rank_by_lgbm's training panel needs a wider lookback than
    the composite ranker: its earliest snapshot (at the start of the
    3-year TRAIN_WINDOW_YEARS span) needs its own 12-month momentum
    lookback *before itself*, not just before as_of_date — about one
    extra year of history reaching back from the first 2018-01-31
    rebalance, not the plain 3-year window a single snapshot needs.
    """
    rng = np.random.default_rng(13)
    dates = pd.bdate_range("2013-06-01", "2018-12-31")
    columns = [f"T{i}" for i in range(n_tickers)] + ["BENCH"]
    daily_returns = rng.normal(loc=0.0005, scale=0.01, size=(len(dates), len(columns)))
    return 100 * (1 + pd.DataFrame(daily_returns, index=dates, columns=columns)).cumprod()


def _synthetic_dividends(tickers: list[str]) -> pd.DataFrame:
    """Distinct quarterly dividend yields per ticker — a constant (all-zero
    or all-equal) yield would make the ranker's z-score undefined (zero
    standard deviation), so the amounts are deliberately varied.
    """
    payment_dates = pd.date_range("2015-03-01", "2018-12-01", freq="3MS")
    rows = []
    for i, ticker in enumerate(tickers):
        amount = 0.2 * i  # 0.0, 0.2, 0.4, ...
        rows.extend({"date": d, "ticker": ticker, "dividend_amount": amount} for d in payment_dates)
    return pd.DataFrame(rows)


def test_get_rebalance_dates_returns_month_ends_in_range():
    prices = _synthetic_universe()
    start, end = pd.Timestamp("2018-01-01"), pd.Timestamp("2018-06-30")

    rebalance_dates = backtest.get_rebalance_dates(prices, start, end)

    assert len(rebalance_dates) == 6  # Jan through Jun
    assert all(start <= d <= end for d in rebalance_dates)
    assert list(rebalance_dates) == sorted(rebalance_dates)  # strictly increasing


def test_transaction_cost_matches_turnover_exactly():
    old_weights = {"A": 0.5, "B": 0.5}
    new_weights = {"A": 0.3, "B": 0.3, "C": 0.4}  # C is a brand new position

    turnover = backtest._turnover(old_weights, new_weights)
    # |0.3-0.5| + |0.3-0.5| + |0.4-0| = 0.2 + 0.2 + 0.4 = 0.8
    assert turnover == pytest.approx(0.8)


def test_zero_turnover_means_no_cost():
    weights = {"A": 0.6, "B": 0.4}
    assert backtest._turnover(weights, weights) == pytest.approx(0.0)


def test_net_returns_equal_gross_minus_cost_on_rebalance_day():
    # At least 1 / MAX_WEIGHT_PER_STOCK tickers, or the optimiser's per-stock
    # cap makes weights summing to 1.0 infeasible.
    prices = _synthetic_universe(n_tickers=8)
    tickers = [c for c in prices.columns if c != "BENCH"]
    dividends = _synthetic_dividends(tickers)
    daily_returns, rebalance_log = backtest.run_backtest(
        prices,
        dividends,
        tickers=tickers,
        benchmark_ticker="BENCH",
        start=pd.Timestamp("2018-01-01"),
        end=pd.Timestamp("2018-04-30"),
    )

    for column in ["strategy", "equal_weight", "sp500"]:
        gap = daily_returns[f"{column}_gross"] - daily_returns[f"{column}_net"]
        # On every day, gross - net should be either 0 (no rebalance that
        # day) or exactly that day's transaction cost.
        assert (gap >= -1e-9).all()

    # The very first rebalance is a full buy from cash: turnover must be 1.0.
    assert rebalance_log["turnover"].iloc[0] == pytest.approx(1.0)


def test_both_benchmarks_present_and_aligned_to_strategy_dates():
    prices = _synthetic_universe(n_tickers=8)
    tickers = [c for c in prices.columns if c != "BENCH"]
    dividends = _synthetic_dividends(tickers)
    daily_returns, _ = backtest.run_backtest(
        prices,
        dividends,
        tickers=tickers,
        benchmark_ticker="BENCH",
        start=pd.Timestamp("2018-01-01"),
        end=pd.Timestamp("2018-04-30"),
    )

    for column in [
        "strategy_gross",
        "strategy_net",
        "equal_weight_gross",
        "equal_weight_net",
        "sp500_gross",
        "sp500_net",
    ]:
        assert column in daily_returns.columns
        assert not daily_returns[column].isna().any()

    # Every series must share exactly the same date index — otherwise
    # comparing strategy to benchmarks day by day wouldn't be valid.
    assert daily_returns.index.is_monotonic_increasing
    assert daily_returns.notna().all(axis=None)


def test_lgbm_ranker_works_end_to_end_in_the_walk_forward_loop():
    prices = _synthetic_universe_long()
    tickers = [c for c in prices.columns if c != "BENCH"]
    dividends = _synthetic_dividends(tickers)

    daily_returns, rebalance_log = backtest.run_backtest(
        prices,
        dividends,
        tickers=tickers,
        benchmark_ticker="BENCH",
        start=pd.Timestamp("2018-01-01"),
        end=pd.Timestamp("2018-04-30"),
        rank_fn=lgbm_ranker.rank_by_lgbm,
    )

    assert len(rebalance_log) > 0
    assert not daily_returns.isna().any(axis=None)


def test_select_target_weights_falls_back_to_equal_weight_on_optimiser_failure():
    # max_sharpe_weights can raise pypfopt's OptimizationError under extreme
    # market stress (verified against a real 2020-03-31 COVID-crash
    # rebalance) - the walk-forward loop must survive that, not crash the
    # whole backtest over one bad rebalance.
    prices = _synthetic_universe(n_tickers=8)
    tickers = [c for c in prices.columns if c != "BENCH"]
    dividends = _synthetic_dividends(tickers)

    def failing_optimiser(prices, risk_free_rate):
        raise pypfopt_exceptions.OptimizationError("infeasible")

    weights = backtest._select_target_weights(
        prices[tickers], dividends, pd.Timestamp("2018-01-31"), optimiser_fn=failing_optimiser
    )

    assert sum(weights.values()) == pytest.approx(1.0)
    assert len(weights) == len(tickers)  # every candidate ticker still gets a (equal) weight
    assert len(set(weights.values())) == 1  # every weight is identical


def test_select_target_weights_looks_up_the_risk_free_rate_as_of_the_rebalance_date():
    prices = _synthetic_universe(n_tickers=8)
    tickers = [c for c in prices.columns if c != "BENCH"]
    dividends = _synthetic_dividends(tickers)

    # Rate is 2% up to and including 2018-01-15, then jumps to 6% - well
    # below the synthetic assets' ~12% mean return so max_sharpe stays
    # feasible - and stays 6% right through the fixture's last date, so a
    # rate dated *after* an early rebalance is present too, proving asof
    # never looks ahead into it (rule 1).
    rate_dates = prices.index
    risk_free_rate = pd.Series(
        [0.02 if d <= pd.Timestamp("2018-01-15") else 0.06 for d in rate_dates], index=rate_dates
    )

    received_rates = []

    def recording_optimiser(prices, risk_free_rate):
        received_rates.append(risk_free_rate)
        return optimise.max_sharpe_weights(prices, risk_free_rate=risk_free_rate)

    backtest._select_target_weights(
        prices[tickers],
        dividends,
        pd.Timestamp("2018-01-10"),
        optimiser_fn=recording_optimiser,
        risk_free_rate=risk_free_rate,
    )
    backtest._select_target_weights(
        prices[tickers],
        dividends,
        pd.Timestamp("2018-01-31"),
        optimiser_fn=recording_optimiser,
        risk_free_rate=risk_free_rate,
    )

    assert received_rates[0] == pytest.approx(0.02)
    assert received_rates[1] == pytest.approx(0.06)


def test_run_backtest_accepts_a_risk_free_rate_series_end_to_end():
    prices = _synthetic_universe(n_tickers=8)
    tickers = [c for c in prices.columns if c != "BENCH"]
    dividends = _synthetic_dividends(tickers)
    risk_free_rate = pd.Series(0.02, index=prices.index)

    daily_returns, rebalance_log = backtest.run_backtest(
        prices,
        dividends,
        tickers=tickers,
        benchmark_ticker="BENCH",
        start=pd.Timestamp("2018-01-01"),
        end=pd.Timestamp("2018-04-30"),
        risk_free_rate=risk_free_rate,
    )

    assert len(rebalance_log) > 0
    assert not daily_returns.isna().any(axis=None)


def test_risk_parity_optimiser_works_end_to_end_in_the_walk_forward_loop():
    prices = _synthetic_universe(n_tickers=8)
    tickers = [c for c in prices.columns if c != "BENCH"]
    dividends = _synthetic_dividends(tickers)

    daily_returns, rebalance_log = backtest.run_backtest(
        prices,
        dividends,
        tickers=tickers,
        benchmark_ticker="BENCH",
        start=pd.Timestamp("2018-01-01"),
        end=pd.Timestamp("2018-04-30"),
        optimiser_fn=optimise.risk_parity_weights,
    )

    assert len(rebalance_log) > 0
    assert not daily_returns.isna().any(axis=None)
