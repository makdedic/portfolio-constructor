"""Guards the LightGBM training panel's lookahead safety (rule 1) and
rolling-window sizing (rule 2) — the trickiest part of this ranker, since
it builds many historical (feature, forward-return) examples rather than
a single snapshot — plus rank_by_lgbm's determinism and output contract.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import features
from src.models import lgbm_ranker
from src.portfolio import backtest


def _synthetic_universe(n_tickers: int = 5) -> pd.DataFrame:
    """~5 years of daily prices — enough for the panel's widened window
    (TRAIN_WINDOW_YEARS + MOMENTUM_LOOKBACK_MONTHS of buffer) plus room
    for an as_of_date comfortably inside the data.
    """
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2014-01-01", "2018-12-31")
    tickers = [f"T{i}" for i in range(n_tickers)]
    daily_returns = rng.normal(loc=0.0005, scale=0.01, size=(len(dates), n_tickers))
    return 100 * (1 + pd.DataFrame(daily_returns, index=dates, columns=tickers)).cumprod()


def _synthetic_dividends(tickers: list[str]) -> pd.DataFrame:
    payment_dates = pd.date_range("2014-03-01", "2018-12-01", freq="3MS")
    rows = []
    for i, ticker in enumerate(tickers):
        amount = 0.2 * i
        rows.extend({"date": d, "ticker": ticker, "dividend_amount": amount} for d in payment_dates)
    return pd.DataFrame(rows)


def test_panel_snapshot_dates_are_the_rebalance_grid_minus_its_last_point():
    universe_prices = _synthetic_universe()
    dividends = _synthetic_dividends(list(universe_prices.columns))
    as_of_date = pd.Timestamp("2018-01-31")

    panel = lgbm_ranker._build_training_panel(universe_prices, dividends, as_of_date)

    # Mirror what the panel builder does internally: as_of_date itself must
    # be excluded before the grid is built, or "the last trading day of the
    # final month" comes out one day later than the panel actually used.
    standard_window_start = as_of_date - pd.DateOffset(years=config.TRAIN_WINDOW_YEARS)
    price_window = features.training_window(
        universe_prices, as_of_date, extra_months=config.MOMENTUM_LOOKBACK_MONTHS
    )
    full_grid = backtest.get_rebalance_dates(price_window, standard_window_start, as_of_date)

    assert set(panel["snapshot_date"].unique()) == set(full_grid[:-1])

    # And the label for the last usable snapshot matches an independently
    # computed forward return, catching sign/alignment bugs separately
    # from date-selection bugs.
    last_snapshot, next_date = full_grid[-2], full_grid[-1]
    expected_label = universe_prices.loc[next_date] / universe_prices.loc[last_snapshot] - 1
    expected_label.index.name = "ticker"
    actual_label = panel.loc[panel["snapshot_date"] == last_snapshot].set_index("ticker")["forward_return"]
    pd.testing.assert_series_equal(expected_label.rename("forward_return"), actual_label, check_like=True)


def test_panel_ignores_price_data_on_as_of_date_itself():
    universe_prices = _synthetic_universe()
    dividends = _synthetic_dividends(list(universe_prices.columns))
    as_of_date = pd.Timestamp("2018-01-31")  # a real business day in the synthetic index
    assert as_of_date in universe_prices.index

    baseline = lgbm_ranker._build_training_panel(universe_prices, dividends, as_of_date)

    mutated = universe_prices.copy()
    mutated.loc[as_of_date] *= 100  # only as_of_date's own row — not a broader future range

    result = lgbm_ranker._build_training_panel(mutated, dividends, as_of_date)
    pd.testing.assert_frame_equal(baseline, result)


def test_panel_builder_raises_a_clear_error_when_history_is_too_short():
    as_of_date = pd.Timestamp("2018-01-31")
    # History starts exactly at the *standard* window boundary, with none of
    # the extra buffer the earliest snapshot's momentum lookback needs.
    standard_window_start = as_of_date - pd.DateOffset(years=config.TRAIN_WINDOW_YEARS)
    dates = pd.bdate_range(standard_window_start, as_of_date - pd.Timedelta(days=1))
    prices = pd.DataFrame({"A": 100.0, "B": 100.0}, index=dates)
    dividends = pd.DataFrame(columns=["date", "ticker", "dividend_amount"])

    with pytest.raises(ValueError, match="[Nn]ot enough"):
        lgbm_ranker._build_training_panel(prices, dividends, as_of_date)


def test_rank_by_lgbm_is_deterministic():
    universe_prices = _synthetic_universe()
    dividends = _synthetic_dividends(list(universe_prices.columns))
    as_of_date = pd.Timestamp("2018-01-31")

    first = lgbm_ranker.rank_by_lgbm(universe_prices, dividends, as_of_date)
    second = lgbm_ranker.rank_by_lgbm(universe_prices, dividends, as_of_date)

    pd.testing.assert_series_equal(first, second)


def test_rank_by_lgbm_matches_score_stocks_output_contract():
    universe_prices = _synthetic_universe()
    dividends = _synthetic_dividends(list(universe_prices.columns))
    as_of_date = pd.Timestamp("2018-01-31")

    scores = lgbm_ranker.rank_by_lgbm(universe_prices, dividends, as_of_date)

    # Same shape as ranker.score_stocks's output: a numeric Series covering
    # every ticker in the universe, so select_top_n works unchanged
    # regardless of which ranker produced it.
    assert isinstance(scores, pd.Series)
    assert set(scores.index) == set(universe_prices.columns)
    assert not scores.isna().any()
