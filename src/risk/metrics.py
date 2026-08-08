"""Risk metrics computed on a daily portfolio-return series.

rule 6: never report raw returns alone. Every metric here answers
a different question about risk, and a bank/fund risk desk would expect all
of them, not just average return.
"""

import numpy as np
import pandas as pd

from src import config


def _daily_excess_returns(
    daily_returns: pd.Series,
    risk_free_rate_annual: float | pd.Series,
    periods_per_year: int,
) -> pd.Series:
    """Subtracts the risk-free rate from daily returns.

    risk_free_rate_annual can be a flat rate (the historical default) or a
    date-indexed Series of actual rates (e.g. from FRED) — reindexed and
    forward-filled onto daily_returns' own dates first, so a real rate that
    only has values on business days still lines up with every trading day.
    """
    if isinstance(risk_free_rate_annual, pd.Series):
        risk_free_rate_annual = risk_free_rate_annual.reindex(daily_returns.index).ffill()
    daily_risk_free_rate = risk_free_rate_annual / periods_per_year
    return daily_returns - daily_risk_free_rate


def sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate_annual: float | pd.Series = config.RISK_FREE_RATE_ANNUAL,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised return per unit of total volatility, in excess of the risk-free rate."""
    excess_returns = _daily_excess_returns(daily_returns, risk_free_rate_annual, periods_per_year)
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)


def sortino_ratio(
    daily_returns: pd.Series,
    risk_free_rate_annual: float | pd.Series = config.RISK_FREE_RATE_ANNUAL,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> float:
    """Like Sharpe, but only penalises downside volatility.

    Upside swings aren't "risk" in the way a Sortino ratio defines it, so
    the denominator only uses the standard deviation of negative excess
    returns.
    """
    excess_returns = _daily_excess_returns(daily_returns, risk_free_rate_annual, periods_per_year)
    downside_returns = excess_returns[excess_returns < 0]
    downside_deviation = downside_returns.std()
    return (excess_returns.mean() / downside_deviation) * np.sqrt(periods_per_year)


def max_drawdown(daily_returns: pd.Series) -> float:
    """The largest peak-to-trough decline in cumulative value.

    Answers "what's the worst loss an investor sitting through the whole
    period would have seen from their own personal high point", which
    average returns and volatility don't capture on their own.
    """
    cumulative_value = (1 + daily_returns).cumprod()
    running_peak = cumulative_value.cummax()
    drawdown = (cumulative_value - running_peak) / running_peak
    return drawdown.min()


def historical_var(daily_returns: pd.Series, confidence: float) -> float:
    """Historical Value at Risk: the loss not expected to be exceeded on
    (1 - confidence) of days, read directly off the empirical return
    distribution (no assumption about its shape).
    """
    return daily_returns.quantile(1 - confidence)


def monte_carlo_var(
    daily_returns: pd.Series,
    confidence: float,
    n_simulations: int = config.MONTE_CARLO_SIMULATIONS,
    seed: int = config.RANDOM_SEED,
) -> float:
    """Monte Carlo Value at Risk: fit a normal distribution to historical
    returns, then simulate many draws from it and read off the same
    quantile.

    Unlike historical_var, this assumes returns are normally distributed —
    the methodological trade-off is a smoother tail estimate in exchange for
    that assumption possibly not holding (real returns have fatter tails).
    """
    rng = np.random.default_rng(seed)
    simulated_returns = rng.normal(daily_returns.mean(), daily_returns.std(), n_simulations)
    return np.quantile(simulated_returns, 1 - confidence)


def expected_shortfall(daily_returns: pd.Series, confidence: float) -> float:
    """Expected Shortfall (CVaR): the average loss on the days that breach VaR.

    VaR only answers "how bad is the threshold" — CVaR answers "given that
    the threshold was breached, how bad was it on average", which is the
    more informative number in a genuine tail event.
    """
    var_threshold = historical_var(daily_returns, confidence)
    return daily_returns[daily_returns <= var_threshold].mean()


def var_breaches(daily_returns: pd.Series, confidence: float) -> int:
    """Count of days where the realised loss exceeded the stated historical VaR.

    This is literally how banks backtest their own VaR models: if a 95% VaR
    is breached on far more than 5% of days, the model is understating risk.
    """
    var_threshold = historical_var(daily_returns, confidence)
    return int((daily_returns < var_threshold).sum())


def compute_all(
    daily_returns: pd.Series,
    risk_free_rate_annual: float | pd.Series = config.RISK_FREE_RATE_ANNUAL,
) -> dict[str, float]:
    """Every risk metric this project reports, keyed by name."""
    metrics = {
        "sharpe_ratio": sharpe_ratio(daily_returns, risk_free_rate_annual),
        "sortino_ratio": sortino_ratio(daily_returns, risk_free_rate_annual),
        "max_drawdown": max_drawdown(daily_returns),
    }
    for confidence in config.VAR_CONFIDENCE_LEVELS:
        confidence_pct = int(confidence * 100)
        metrics[f"historical_var_{confidence_pct}"] = historical_var(daily_returns, confidence)
        metrics[f"monte_carlo_var_{confidence_pct}"] = monte_carlo_var(daily_returns, confidence)
        metrics[f"expected_shortfall_{confidence_pct}"] = expected_shortfall(
            daily_returns, confidence
        )
        metrics[f"var_breaches_{confidence_pct}"] = var_breaches(daily_returns, confidence)
    return metrics
