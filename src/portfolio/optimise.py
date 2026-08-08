"""Portfolio optimisation with Ledoit-Wolf covariance shrinkage: mean-variance
max Sharpe (return-seeking) and risk parity (risk-balancing). backtest.py
calls this module through an `optimiser_fn` parameter (see
`_select_target_weights` and `run_backtest`), so both functions share the
same `(prices) -> dict[str, float]` contract and are interchangeable there.
"""

import cvxpy as cp
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
    covariance matrix is noisy, and shrinkage pulls it towards a more stable,
    structured estimate. Never break this rule (rule 4) — there is
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


def _cap_and_redistribute(weights: dict[str, float], max_weight: float) -> dict[str, float]:
    """Clip any weight above max_weight down to it, redistributing the
    excess proportionally among the weights still below the cap. A capped
    weight never receives more, so this repeats until nothing exceeds the
    cap (usually converges in one or two passes).
    """
    weights = dict(weights)
    while max(weights.values()) > max_weight + 1e-9:
        uncapped = {ticker: w for ticker, w in weights.items() if w < max_weight}
        excess = sum(w - max_weight for w in weights.values() if w >= max_weight)
        uncapped_total = sum(uncapped.values())
        weights = {
            ticker: (max_weight if w >= max_weight else w + excess * (w / uncapped_total))
            for ticker, w in weights.items()
        }
    return weights


def risk_parity_weights(prices: pd.DataFrame, risk_free_rate: float | None = None) -> dict[str, float]:
    """Weights such that every holding contributes equally to total
    portfolio risk (Equal Risk Contribution), rather than being weighted
    towards the best *estimated* returns the way max_sharpe_weights is —
    return estimates are noisy, so risk parity sidesteps them entirely and
    balances risk contributions instead.

    risk_free_rate is accepted but unused: it only exists so this function
    matches max_sharpe_weights' call signature for backtest.py's pluggable
    optimiser_fn contract. ERC never references expected returns, so there
    is nothing for a risk-free rate to do here.

    Solved via the standard convex reformulation (Maillard, Roncalli &
    Teiletche, 2010): minimising 0.5 * w'Σw - sum(ln(w_i)) over w > 0 (no
    other constraint), then renormalising the result to sum to 1, gives
    exactly the ERC solution — that objective's optimality condition works
    out to w_i * (Σw)_i being equal for every i.

    MAX_WEIGHT_PER_STOCK is applied as a *separate* step after solving and
    renormalising, not as a solver constraint on the raw variable: the raw
    variable's scale before renormalising is arbitrary (only its direction
    matters), so bounding it directly bounds an arbitrary, meaningless
    number — in testing this produced every weight pinned to exactly the
    bound, a plausible-looking but wrong "solution". Capping the properly
    normalised weights and redistributing the excess among the rest avoids
    that entirely.

    Ledoit-Wolf shrinkage (rule 4) — no code path here uses raw sample
    covariance, same as max_sharpe_weights.
    """
    shrunk_covariance = risk_models.CovarianceShrinkage(
        prices, frequency=config.TRADING_DAYS_PER_YEAR
    ).ledoit_wolf()
    tickers = list(shrunk_covariance.columns)
    covariance_matrix = shrunk_covariance.values

    weights = cp.Variable(len(tickers))
    portfolio_risk = cp.quad_form(weights, covariance_matrix)
    objective = cp.Minimize(0.5 * portfolio_risk - cp.sum(cp.log(weights)))
    problem = cp.Problem(objective, [weights >= 1e-6])
    problem.solve()

    normalised_weights = weights.value / weights.value.sum()
    equal_risk_weights = dict(zip(tickers, normalised_weights))
    return _cap_and_redistribute(equal_risk_weights, config.MAX_WEIGHT_PER_STOCK)
