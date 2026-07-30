"""Guards rule 6: hand-verifiable correctness for each risk metric, plus the
general relationships that must always hold between them.
"""

import numpy as np
import pandas as pd
import pytest

from src.risk import metrics


def test_sharpe_ratio_matches_manual_calculation():
    daily_returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])
    expected = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    result = metrics.sharpe_ratio(daily_returns, risk_free_rate_annual=0.0, periods_per_year=252)
    assert result == pytest.approx(expected)


def test_sortino_penalises_only_downside_and_exceeds_sharpe_here():
    daily_returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])
    downside = daily_returns[daily_returns < 0]
    expected = (daily_returns.mean() / downside.std()) * np.sqrt(252)

    sortino = metrics.sortino_ratio(daily_returns, risk_free_rate_annual=0.0, periods_per_year=252)
    sharpe = metrics.sharpe_ratio(daily_returns, risk_free_rate_annual=0.0, periods_per_year=252)

    assert sortino == pytest.approx(expected)
    # Downside deviation here is smaller than total deviation (the up moves
    # are bigger than the down moves), so Sortino must exceed Sharpe.
    assert sortino > sharpe


def test_sharpe_and_sortino_use_a_time_varying_risk_free_rate():
    dates = pd.bdate_range("2020-01-01", periods=10)
    daily_returns = pd.Series(
        [0.01, -0.01, 0.02, -0.02, 0.01, 0.015, -0.005, 0.02, -0.015, 0.01], index=dates
    )
    # Rate jumps from 0% to 10% halfway through - if the per-day rate isn't
    # actually being subtracted day by day, this wouldn't match a manual
    # calculation that does use the real per-day value.
    risk_free_rate = pd.Series([0.0] * 5 + [0.10] * 5, index=dates)
    expected_excess = daily_returns - (risk_free_rate / 252)

    expected_sharpe = (expected_excess.mean() / expected_excess.std()) * np.sqrt(252)
    sharpe = metrics.sharpe_ratio(daily_returns, risk_free_rate_annual=risk_free_rate, periods_per_year=252)
    assert sharpe == pytest.approx(expected_sharpe)

    downside = expected_excess[expected_excess < 0]
    expected_sortino = (expected_excess.mean() / downside.std()) * np.sqrt(252)
    sortino = metrics.sortino_ratio(daily_returns, risk_free_rate_annual=risk_free_rate, periods_per_year=252)
    assert sortino == pytest.approx(expected_sortino)


def test_time_varying_risk_free_rate_is_forward_filled_onto_return_dates():
    dates = pd.bdate_range("2020-01-01", periods=6)
    daily_returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01, 0.015], index=dates)
    # Sparse rate series, e.g. FRED's own publishing gaps - only has a
    # value on the first and fourth day of the return series.
    sparse_rate = pd.Series([0.02, 0.06], index=[dates[0], dates[3]])

    filled_rate = sparse_rate.reindex(dates).ffill()
    expected_excess = daily_returns - (filled_rate / 252)
    expected_sharpe = (expected_excess.mean() / expected_excess.std()) * np.sqrt(252)

    sharpe = metrics.sharpe_ratio(daily_returns, risk_free_rate_annual=sparse_rate, periods_per_year=252)
    assert sharpe == pytest.approx(expected_sharpe)


def test_max_drawdown_known_series():
    # Cumulative value: 1.10, 0.88, 0.924, 1.0164 — the worst drop from the
    # running peak of 1.10 is at the second point: (0.88 - 1.10) / 1.10 = -0.2.
    daily_returns = pd.Series([0.10, -0.20, 0.05, 0.10])
    assert metrics.max_drawdown(daily_returns) == pytest.approx(-0.2)


def test_historical_var_and_dependent_metrics_on_a_known_distribution():
    daily_returns = pd.Series(range(1, 101), dtype=float)
    confidence = 0.95

    var_95 = metrics.historical_var(daily_returns, confidence)
    assert var_95 == pytest.approx(daily_returns.quantile(1 - confidence))

    expected_shortfall = metrics.expected_shortfall(daily_returns, confidence)
    assert expected_shortfall == pytest.approx(daily_returns[daily_returns <= var_95].mean())

    breaches = metrics.var_breaches(daily_returns, confidence)
    assert breaches == (daily_returns < var_95).sum()
    # By construction, roughly (1 - confidence) of observations should breach.
    assert breaches == pytest.approx(100 * (1 - confidence), abs=2)


def test_expected_shortfall_is_never_milder_than_var():
    daily_returns = pd.Series(np.random.default_rng(0).normal(0, 0.01, 1_000))
    for confidence in [0.95, 0.99]:
        var = metrics.historical_var(daily_returns, confidence)
        expected_shortfall = metrics.expected_shortfall(daily_returns, confidence)
        # CVaR is the average of the tail beyond (and including) VaR, so it
        # can never be a smaller loss than the VaR threshold itself.
        assert expected_shortfall <= var


def test_monte_carlo_var_is_reproducible_given_the_same_seed():
    daily_returns = pd.Series(np.random.default_rng(1).normal(0, 0.01, 500))
    first = metrics.monte_carlo_var(daily_returns, confidence=0.95, seed=42)
    second = metrics.monte_carlo_var(daily_returns, confidence=0.95, seed=42)
    assert first == second


def test_compute_all_includes_every_required_metric():
    daily_returns = pd.Series(np.random.default_rng(2).normal(0.0003, 0.01, 300))
    result = metrics.compute_all(daily_returns)

    expected_keys = {
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "historical_var_95",
        "monte_carlo_var_95",
        "expected_shortfall_95",
        "var_breaches_95",
        "historical_var_99",
        "monte_carlo_var_99",
        "expected_shortfall_99",
        "var_breaches_99",
    }
    assert expected_keys.issubset(result.keys())
