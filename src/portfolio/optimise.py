"""Mean-variance portfolio optimisation with Ledoit-Wolf covariance shrinkage.

Risk-parity optimisation is deferred to a later pass; backtest.py calls this
module through an `optimisation_method` parameter so adding risk-parity
later doesn't require changing the backtest loop's structure.
"""

import pandas as pd
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

from src import config


def max_sharpe_weights(
    prices: pd.DataFrame, risk_free_rate: float = config.RISK_FREE_RATE_ANNUAL
) -> dict[str, float]:
    """Mean-variance weights that maximise the Sharpe ratio.

    Covariance is estimated with Ledoit-Wolf shrinkage rather than the raw
    sample covariance: with only a few years of daily data, the raw sample
    covariance matrix is noisy, and shrinkage pulls it toward a more stable,
    structured estimate. Never break this rule (claude.md rule 4) — there is
    no code path here using raw sample covariance.
    """
    expected_annual_returns = expected_returns.mean_historical_return(
        prices, frequency=config.TRADING_DAYS_PER_YEAR
    )
    shrunk_covariance = risk_models.CovarianceShrinkage(
        prices, frequency=config.TRADING_DAYS_PER_YEAR
    ).ledoit_wolf()

    frontier = EfficientFrontier(
        expected_annual_returns, shrunk_covariance, weight_bounds=(0, config.MAX_WEIGHT_PER_STOCK)
    )
    frontier.max_sharpe(risk_free_rate=risk_free_rate)
    return frontier.clean_weights()
