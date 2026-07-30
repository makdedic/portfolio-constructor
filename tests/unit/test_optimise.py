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


def _synthetic_prices_varied_risk(n_tickers: int = 12) -> pd.DataFrame:
    """Deliberately different volatility per ticker, unlike _synthetic_prices'
    near-identical assets. Equal risk contribution collapses to equal weight
    by symmetry when every asset has similar risk, which wouldn't
    distinguish a correct implementation from a bug that happens to also
    produce equal weights regardless of actual risk (which is exactly what
    happened once during development) - this fixture is what makes the
    property test below a real, discriminating check.

    The volatility spread (2x, not more) and ticker count are deliberately
    modest: too wide a spread with too few tickers pushes the natural ERC
    weight on the lowest-volatility ticker above MAX_WEIGHT_PER_STOCK,
    which caps it — correct behaviour, but it would test the capping logic
    instead of the pure equal-risk-contribution property this test wants.
    """
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2020-01-01", "2022-12-31")
    tickers = [f"T{i}" for i in range(n_tickers)]
    volatilities = np.linspace(0.010, 0.020, n_tickers)
    daily_returns = rng.normal(loc=0.0003, scale=volatilities, size=(len(dates), n_tickers))
    return 100 * (1 + pd.DataFrame(daily_returns, index=dates, columns=tickers)).cumprod()


def test_risk_parity_achieves_equal_risk_contribution():
    prices = _synthetic_prices_varied_risk()
    weights = optimise.risk_parity_weights(prices)

    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])
    shrunk_covariance = risk_models.CovarianceShrinkage(
        prices, frequency=config.TRADING_DAYS_PER_YEAR
    ).ledoit_wolf()
    covariance_matrix = shrunk_covariance.loc[tickers, tickers].values

    risk_contributions = w * (covariance_matrix @ w)
    assert risk_contributions.std() == pytest.approx(0, abs=1e-6)

    # Confirm the fixture's assets genuinely have unequal risk, so an
    # equal-weight portfolio would *not* have equal risk contribution -
    # otherwise this test couldn't actually distinguish a correct ERC
    # solve from a bug that happens to also produce equal weights.
    equal_weight = np.full(len(tickers), 1 / len(tickers))
    equal_weight_contributions = equal_weight * (covariance_matrix @ equal_weight)
    assert equal_weight_contributions.std() > 1e-6


def test_risk_parity_weights_sum_to_one_and_respect_bounds():
    n_tickers = int(np.ceil(1 / config.MAX_WEIGHT_PER_STOCK)) + 1
    prices = _synthetic_prices(n_tickers)
    weights = optimise.risk_parity_weights(prices)

    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(0 <= weight <= config.MAX_WEIGHT_PER_STOCK + 1e-9 for weight in weights.values())
