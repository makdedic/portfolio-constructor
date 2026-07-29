"""LightGBM-based stock ranker: predicts next-month return per ticker,
trained fresh at every rebalance on a rolling panel of historical examples.

Every example has the same three inputs (features) as the composite
ranker in ranker.py — momentum, low_volatility, dividend_yield — paired
with what forward_return turned out to be historically (the label: the
realized return over the following month, only ever knowable in
hindsight). Training teaches the model a relationship between those three
inputs and what actually happened next; predicting for the real, current
as_of_date feeds it the same three inputs (never forward_return, which
doesn't exist yet for the future) and reads off its predicted return as
the ranking score.

Never a model trained once on full history (rule 2), and never on a
training example whose forward return would require data at or after the
prediction date (rule 1).
"""

import pandas as pd
from lightgbm import LGBMRegressor

from src import config
from src.data import features
from src.portfolio import backtest

_FEATURE_COLUMNS = ["momentum", "low_volatility", "dividend_yield"]


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
    window: the earliest candidate snapshot needs its own 12 months of
    momentum lookback *before* the window starts, which a plain
    TRAIN_WINDOW_YEARS slice doesn't contain. That extra room is a data
    buffer, not extra training examples — the snapshot dates themselves
    are still restricted to the true TRAIN_WINDOW_YEARS span.

    The buffer is MOMENTUM_LOOKBACK_MONTHS plus one extra month of slack:
    exactly MOMENTUM_LOOKBACK_MONTHS lines up in theory, but weekends and
    leap-year Feb 29 lookbacks (which pandas clips to Feb 28 the following
    non-leap year) round the two boundaries in different directions by a
    few days often enough that the exact buffer isn't quite enough.
    """
    as_of_ts = pd.Timestamp(as_of_date)
    panel_price_window = features.training_window(
        universe_prices, as_of_ts, extra_months=config.MOMENTUM_LOOKBACK_MONTHS + 1
    )
    standard_window_start = as_of_ts - pd.DateOffset(years=config.TRAIN_WINDOW_YEARS)

    snapshot_grid = backtest.get_rebalance_dates(panel_price_window, standard_window_start, as_of_ts)
    usable_snapshots = snapshot_grid[:-1]  # the last one has no valid forward-return label

    panel_rows = []
    for i, snapshot_date in enumerate(usable_snapshots):
        next_date = snapshot_grid[i + 1]

        # A date-arithmetic pre-check here is too fragile to get right (the
        # expected boundary can fall on a weekend/holiday and shift by a
        # day or two, same as everywhere else prices are looked up in this
        # project) — instead, catch the actual failure if the earliest
        # snapshot genuinely doesn't have enough lookback behind it.
        try:
            momentum = features.compute_momentum(panel_price_window, snapshot_date)
            low_volatility = features.compute_low_volatility(panel_price_window, snapshot_date)
            dividend_yield = features.compute_dividend_yield(dividends, panel_price_window, snapshot_date)
        except IndexError as error:
            raise ValueError(
                f"Not enough price history to compute features for snapshot "
                f"{snapshot_date.date()} (as_of_date={as_of_ts.date()}): {error}"
            ) from error
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


def rank_by_lgbm(universe_prices: pd.DataFrame, dividends: pd.DataFrame, as_of_date) -> pd.Series:
    """Predict next-month return per ticker with a LightGBM model trained
    fresh on this call's own training panel.

    Same (universe_prices, dividends, as_of_date) -> scores contract as
    ranker.rank_by_composite_score, so it's a drop-in rank_fn for
    backtest.run_backtest.
    """
    training_panel = _build_training_panel(universe_prices, dividends, as_of_date)

    model = LGBMRegressor(
        n_estimators=config.LGBM_N_ESTIMATORS,
        max_depth=config.LGBM_MAX_DEPTH,
        learning_rate=config.LGBM_LEARNING_RATE,
        min_child_samples=config.LGBM_MIN_CHILD_SAMPLES,
        random_state=config.RANDOM_SEED,
        n_jobs=1,
        deterministic=True,
        verbose=-1,
    )
    model.fit(training_panel[_FEATURE_COLUMNS], training_panel["forward_return"])

    # The current snapshot uses the standard (unwidened) window, same as
    # every other ranker — only the training panel's earliest rows needed
    # the extra lookback buffer.
    current_window = features.training_window(universe_prices, as_of_date)
    current_features = pd.DataFrame(
        {
            "momentum": features.compute_momentum(current_window, as_of_date),
            "low_volatility": features.compute_low_volatility(current_window, as_of_date),
            "dividend_yield": features.compute_dividend_yield(dividends, current_window, as_of_date),
        }
    )

    predicted_returns = model.predict(current_features[_FEATURE_COLUMNS])
    return pd.Series(predicted_returns, index=current_features.index)
