"""Guards rule 1 (no lookahead bias): a factor computed as of date t must be
unchanged by any data dated t or later.
"""

import pandas as pd

from src.data import features

AS_OF = pd.Timestamp("2021-06-01")


def _synthetic_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", "2022-06-30")
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
