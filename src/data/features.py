"""Alpha factors: signals believed to predict which stocks will outperform.

Every function takes an `as_of_date` and only ever looks at data strictly
before it — that one convention (`_as_of`, `_before`) is what satisfies the
no-lookahead-bias rule, and it's the only place that rule is enforced.
"""

import pandas as pd

from src import config


def _as_of(prices: pd.DataFrame, as_of_date) -> pd.DataFrame:
    """Price history strictly before as_of_date."""
    return prices.loc[prices.index < pd.Timestamp(as_of_date)]


def _price_on_or_before(prices: pd.DataFrame, target_date: pd.Timestamp) -> pd.Series:
    """Most recent available price on or before target_date, per ticker.

    target_date rarely lands on an actual trading day (weekends, holidays),
    so this looks back to the last day that traded.
    """
    eligible = prices.loc[prices.index <= target_date]
    return eligible.iloc[-1]


def compute_momentum(prices: pd.DataFrame, as_of_date) -> pd.Series:
    """12-1 month momentum: return from 12 months ago to 1 month ago.

    The most recent month is skipped because short-term price moves tend to
    reverse, while the longer trend tends to continue — this is the
    standard academic momentum factor definition.
    """
    history = _as_of(prices, as_of_date)
    as_of_ts = pd.Timestamp(as_of_date)
    start_price = _price_on_or_before(
        history, as_of_ts - pd.DateOffset(months=config.MOMENTUM_LOOKBACK_MONTHS)
    )
    end_price = _price_on_or_before(
        history, as_of_ts - pd.DateOffset(months=config.MOMENTUM_SKIP_MONTHS)
    )
    return (end_price / start_price) - 1


def compute_low_volatility(prices: pd.DataFrame, as_of_date) -> pd.Series:
    """Trailing 12-month realized daily-return volatility.

    Lower volatility stocks are the "low-vol" factor bet: historically they
    have delivered better risk-adjusted returns than their beta would predict.
    """
    history = _as_of(prices, as_of_date)
    as_of_ts = pd.Timestamp(as_of_date)
    window_start = as_of_ts - pd.DateOffset(months=config.LOW_VOL_LOOKBACK_MONTHS)
    window = history.loc[history.index >= window_start]
    daily_returns = window.pct_change().dropna()
    return daily_returns.std()


def compute_dividend_yield(dividends: pd.DataFrame, prices: pd.DataFrame, as_of_date) -> pd.Series:
    """Trailing 12-month dividend yield: dividends paid, divided by current price.

    This is the "value" factor for this pass: a high yield relative to price
    is a classic (if imperfect) signal that a stock is cheap.
    """
    as_of_ts = pd.Timestamp(as_of_date)
    window_start = as_of_ts - pd.DateOffset(months=config.DIVIDEND_YIELD_LOOKBACK_MONTHS)
    window = dividends.loc[(dividends["date"] >= window_start) & (dividends["date"] < as_of_ts)]
    trailing_dividends = window.groupby("ticker")["dividend_amount"].sum()

    current_price = _price_on_or_before(_as_of(prices, as_of_date), as_of_ts)
    return trailing_dividends.reindex(current_price.index).fillna(0.0) / current_price
