"""Guards rule 1 (no lookahead bias) and rule 2 (rolling window, never full
history): a factor computed as of date t must be unchanged by any data
dated t or later, and training_window must be a fixed-size rolling slice.
"""

import pandas as pd

from src.data import features

AS_OF = pd.Timestamp("2021-06-01")


def _synthetic_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", "2022-06-30")
    trend = pd.Series(range(len(dates)), index=dates, dtype=float)
    return pd.DataFrame({"A": 100 + trend * 0.1, "B": 100 + trend * 0.05})


def _synthetic_prices_long() -> pd.DataFrame:
    """~4 years — long enough to prove training_window is a rolling slice,
    not everything available (the other synthetic series above is too short
    for that distinction to show up).
    """
    dates = pd.bdate_range("2015-01-01", "2018-12-31")
    trend = pd.Series(range(len(dates)), index=dates, dtype=float)
    return pd.DataFrame({"A": 100 + trend * 0.1, "B": 100 + trend * 0.05})


def _synthetic_dividends() -> pd.DataFrame:
    dates = pd.date_range("2020-03-01", "2022-06-01", freq="3MS")
    return pd.DataFrame(
        {"date": dates, "ticker": ["A"] * len(dates), "dividend_amount": [0.5] * len(dates)}
    )


def test_momentum_ignores_data_on_or_after_as_of_date():
    prices = _synthetic_prices()
    baseline = features.compute_momentum(prices, AS_OF)

    mutated = prices.copy()
    mutated.loc[mutated.index >= AS_OF] *= 100  # blow up every future price

    result = features.compute_momentum(mutated, AS_OF)
    pd.testing.assert_series_equal(baseline, result)


def test_low_volatility_ignores_data_on_or_after_as_of_date():
    prices = _synthetic_prices()
    baseline = features.compute_low_volatility(prices, AS_OF)

    mutated = prices.copy()
    mutated.loc[mutated.index >= AS_OF] *= 100

    result = features.compute_low_volatility(mutated, AS_OF)
    pd.testing.assert_series_equal(baseline, result)


def test_dividend_yield_ignores_dividends_on_or_after_as_of_date():
    prices = _synthetic_prices()
    dividends = _synthetic_dividends()
    baseline = features.compute_dividend_yield(dividends, prices, AS_OF)

    mutated = dividends.copy()
    mutated.loc[mutated["date"] >= AS_OF, "dividend_amount"] = 999.0

    result = features.compute_dividend_yield(mutated, prices, AS_OF)
    pd.testing.assert_series_equal(baseline, result)


def test_training_window_is_rolling_never_full_history():
    prices = _synthetic_prices_long()
    rebalance_dates = pd.date_range("2018-01-31", "2018-06-30", freq="ME")

    window_start_dates = []
    for rebalance_date in rebalance_dates:
        window = features.training_window(prices, rebalance_date)

        # Rule 1: nothing in the window is on or after the rebalance date.
        assert window.index.max() < rebalance_date
        # Rule 2: the window is a ~3-year slice, not everything available
        # before the rebalance date (the full history goes back to 2015-01).
        assert window.index.min() > pd.Timestamp("2015-01-01")
        window_start_dates.append(window.index.min())

    # And it actually rolls forward each month rather than staying fixed.
    assert window_start_dates == sorted(window_start_dates)
    assert window_start_dates[0] < window_start_dates[-1]


def test_training_window_extra_months_widens_the_start_only():
    prices = _synthetic_prices_long()
    as_of_date = pd.Timestamp("2018-06-30")

    standard_window = features.training_window(prices, as_of_date)
    widened_window = features.training_window(prices, as_of_date, extra_months=12)

    assert widened_window.index.min() < standard_window.index.min()
    assert widened_window.index.max() == standard_window.index.max()
