"""LightGBM-based stock ranker: predicts next-month return per ticker,
trained fresh at every rebalance on a rolling panel of historical
(features, realized forward return) examples — never a model trained once
on full history (rule 2), and never on a training example whose forward
return would require data at or after the prediction date (rule 1).
"""

import pandas as pd

from src import config
from src.data import features
from src.portfolio import backtest


def _build_training_panel(universe_prices: pd.DataFrame, dividends: pd.DataFrame, as_of_date) -> pd.DataFrame:
    """One row per ticker per usable monthly snapshot before as_of_date.

    Columns: ticker, snapshot_date, momentum, low_volatility,
    dividend_yield, forward_return.

    The snapshot grid is built only from a price window that already
    excludes as_of_date and later (features.training_window's own
    lookahead-safety guarantee), so as_of_date itself can never be a
    candidate snapshot. Given that grid, the *last* snapshot has no valid
    "next" date to compute a forward-return label from — that's the one
    exclusion needed, not two.

    The window feeding this panel is wider than the standard training
    window (extra_months=MOMENTUM_LOOKBACK_MONTHS): the earliest candidate
    snapshot needs its own 12 months of momentum lookback *before* the
    window starts, which a plain TRAIN_WINDOW_YEARS slice doesn't contain.
    That extra room is a data buffer, not extra training examples — the
    snapshot dates themselves are still restricted to the true
    TRAIN_WINDOW_YEARS span.
    """
    as_of_ts = pd.Timestamp(as_of_date)
    panel_price_window = features.training_window(
        universe_prices, as_of_ts, extra_months=config.MOMENTUM_LOOKBACK_MONTHS
    )
    standard_window_start = as_of_ts - pd.DateOffset(years=config.TRAIN_WINDOW_YEARS)

    expected_start = standard_window_start - pd.DateOffset(months=config.MOMENTUM_LOOKBACK_MONTHS)
    if panel_price_window.empty or panel_price_window.index.min() > expected_start:
        earliest_available = panel_price_window.index.min().date() if not panel_price_window.empty else "none"
        raise ValueError(
            f"Not enough price history to build a training panel as of {as_of_ts.date()}: "
            f"need data from {expected_start.date()}, earliest available is {earliest_available}."
        )

    snapshot_grid = backtest.get_rebalance_dates(panel_price_window, standard_window_start, as_of_ts)
    usable_snapshots = snapshot_grid[:-1]  # the last one has no valid forward-return label

    panel_rows = []
    for i, snapshot_date in enumerate(usable_snapshots):
        next_date = snapshot_grid[i + 1]

        momentum = features.compute_momentum(panel_price_window, snapshot_date)
        low_volatility = features.compute_low_volatility(panel_price_window, snapshot_date)
        dividend_yield = features.compute_dividend_yield(dividends, panel_price_window, snapshot_date)
        forward_return = panel_price_window.loc[next_date] / panel_price_window.loc[snapshot_date] - 1

        snapshot_rows = pd.DataFrame(
            {
                "momentum": momentum,
                "low_volatility": low_volatility,
                "dividend_yield": dividend_yield,
                "forward_return": forward_return,
            }
        )
        snapshot_rows["ticker"] = snapshot_rows.index
        snapshot_rows["snapshot_date"] = snapshot_date
        panel_rows.append(snapshot_rows.reset_index(drop=True))

    return pd.concat(panel_rows, ignore_index=True)
