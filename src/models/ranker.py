"""Combines the alpha factors into a single ranking score.

Pass 1 uses a simple composite z-score rather than a trained model. This
keeps the first end-to-end pipeline deterministic and easy to explain, and
it's the exact seam a future LightGBM ranker replaces: same inputs, same
output shape (a ranked Series), nothing downstream has to change.
"""

import pandas as pd

from src import config


def _zscore(factor: pd.Series) -> pd.Series:
    """Standardise a factor to mean 0, std 1.

    Momentum, volatility, and dividend yield are on different scales (a
    return, a standard deviation, a percentage) — z-scoring puts them on a
    common scale so they can be averaged together meaningfully.
    """
    return (factor - factor.mean()) / factor.std()


def score_stocks(
    momentum: pd.Series, low_volatility: pd.Series, dividend_yield: pd.Series
) -> pd.Series:
    """Rank stocks by the average of their z-scored factors.

    Volatility is negated before z-scoring, since a *lower* value should
    score *higher* (the "low-volatility" factor bet), unlike momentum and
    dividend yield where higher is better.
    """
    scored_factors = pd.DataFrame(
        {
            "momentum": _zscore(momentum),
            "low_volatility": _zscore(-low_volatility),
            "dividend_yield": _zscore(dividend_yield),
        }
    )
    return scored_factors.mean(axis=1)


def select_top_n(scores: pd.Series, n: int = config.TOP_N_HOLDINGS) -> pd.Index:
    """Tickers of the n highest-scoring stocks."""
    return scores.sort_values(ascending=False).head(n).index
