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


def training_window(
    prices: pd.DataFrame, as_of_date, years: int = config.TRAIN_WINDOW_YEARS, extra_months: int = 0
) -> pd.DataFrame:
    """Rolling window strictly before as_of_date — the single mechanism
    enforcing both rule 1 (no lookahead) and rule 2 (rolling, never full
    history) for anything computed from it.

    extra_months widens the window's start without moving its end, for
    callers whose earliest usable snapshot needs its own lookback room
    before the window itself starts (e.g. a multi-date training panel,
    rather than a single as-of-date query).
    """
    as_of_ts = pd.Timestamp(as_of_date)
    window_start = as_of_ts - pd.DateOffset(years=years, months=extra_months)
    return prices.loc[(prices.index >= window_start) & (prices.index < as_of_ts)]


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


def tickers_with_complete_history(wide_prices: pd.DataFrame) -> list[str]:
    """Tickers with no missing price anywhere in wide_prices' date range.

    Applied once, up front (e.g. by backtest.run_backtest, right after
    slicing to the requested universe), rather than requiring every
    downstream factor/ranking function to handle partial history
    individually: a ticker too young to have this much history (an IPO or
    spinoff after the range starts) is excluded from the whole backtest,
    same "drop it entirely" simplification used for the full S&P 500
    universe.
    """
    return wide_prices.columns[wide_prices.notna().all()].tolist()
