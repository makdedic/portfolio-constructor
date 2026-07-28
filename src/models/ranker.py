"""Combines the alpha factors into a single ranking score.

`score_stocks` uses a simple composite z-score rather than a trained model
— deterministic and easy to explain. `rank_by_composite_score`/
`rank_by_momentum_only` adapt it to backtest.py's `rank_fn(universe_prices,
dividends, as_of_date) -> pd.Series` contract, which is the seam a
model-based ranker (e.g. LightGBM, which needs the raw data to build its
own training panel rather than a single factor snapshot) plugs into.
"""

import pandas as pd

from src import config
from src.data import features


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


def score_by_momentum_only(
    momentum: pd.Series, low_volatility: pd.Series, dividend_yield: pd.Series
) -> pd.Series:
    """Rank stocks by momentum alone, ignoring low-volatility and dividend yield.

    Takes the same three factors as score_stocks (so the two are
    interchangeable wherever a scoring function is expected), but only uses
    momentum. A useful comparison: pure momentum chases whatever is
    trending hardest — including high-growth, no-dividend, high-volatility
    names that score_stocks structurally avoids — at the cost of being more
    exposed to sharp momentum reversals.
    """
    return _zscore(momentum)


def select_top_n(scores: pd.Series, n: int = config.TOP_N_HOLDINGS) -> pd.Index:
    """Tickers of the n highest-scoring stocks."""
    return scores.sort_values(ascending=False).head(n).index


def rank_by_composite_score(universe_prices: pd.DataFrame, dividends: pd.DataFrame, as_of_date) -> pd.Series:
    """score_stocks, computing its own factors from raw prices/dividends.

    This wider signature — raw data in, scores out — is what makes ranking
    rules interchangeable in backtest.py: score_stocks only needs a single
    snapshot of already-computed factors, but a model-based ranker (e.g. a
    future LightGBM ranker) needs the raw data to build its own training
    panel across many dates, so every rank_fn has to share this shape.
    """
    window = features.training_window(universe_prices, as_of_date)
    momentum = features.compute_momentum(window, as_of_date)
    low_volatility = features.compute_low_volatility(window, as_of_date)
    dividend_yield = features.compute_dividend_yield(dividends, window, as_of_date)
    return score_stocks(momentum, low_volatility, dividend_yield)


def rank_by_momentum_only(universe_prices: pd.DataFrame, dividends: pd.DataFrame, as_of_date) -> pd.Series:
    """score_by_momentum_only, computing its own factors from raw prices/dividends."""
    window = features.training_window(universe_prices, as_of_date)
    momentum = features.compute_momentum(window, as_of_date)
    low_volatility = features.compute_low_volatility(window, as_of_date)
    dividend_yield = features.compute_dividend_yield(dividends, window, as_of_date)
    return score_by_momentum_only(momentum, low_volatility, dividend_yield)
