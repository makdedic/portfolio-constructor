"""Guards rule 4 (Ledoit-Wolf shrinkage) and the optimiser's basic contract."""

import numpy as np
import pandas as pd
import pytest
from pypfopt import risk_models

from src import config
from src.portfolio import optimise


def _synthetic_prices(n_tickers: int = 4) -> pd.DataFrame:
    # Drift comfortably exceeds noise here so every asset's historical mean
    # return is unambiguously positive — max_sharpe requires at least one
    # expected return above the risk-free rate, and this test isn't about
    # realistic market noise, just the optimiser's contract.
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", "2022-12-31")
    tickers = [f"T{i}" for i in range(n_tickers)]
    daily_returns = rng.normal(loc=0.0006, scale=0.008, size=(len(dates), len(tickers)))
    return 100 * (1 + pd.DataFrame(daily_returns, index=dates, columns=tickers)).cumprod()


def test_ledoit_wolf_shrinks_every_covariance_toward_zero():
    prices = _synthetic_prices()

    raw_covariance = risk_models.sample_cov(prices)
    shrunk_covariance = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    off_diagonal_mask = ~np.eye(len(prices.columns), dtype=bool)
    raw_off_diagonal = raw_covariance.to_numpy()[off_diagonal_mask]
    shrunk_off_diagonal = shrunk_covariance.to_numpy()[off_diagonal_mask]

    # Ledoit-Wolf shrinkage pulls every off-diagonal covariance strictly
    # toward zero, so its magnitude must be strictly smaller than the raw
    # sample estimate's — this is rule 4.
    assert np.all(np.abs(shrunk_off_diagonal) < np.abs(raw_off_diagonal))


def test_max_sharpe_weights_sum_to_one_and_respect_bounds():
    # Enough tickers that the MAX_WEIGHT_PER_STOCK cap can still sum to 1.0
    # (fewer tickers than 1 / MAX_WEIGHT_PER_STOCK makes the constraint infeasible.
    n_tickers = int(np.ceil(1 / config.MAX_WEIGHT_PER_STOCK)) + 1
    prices = _synthetic_prices(n_tickers)
    weights = optimise.max_sharpe_weights(prices, risk_free_rate=0.02)

    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(0 <= weight <= config.MAX_WEIGHT_PER_STOCK + 1e-9 for weight in weights.values())
