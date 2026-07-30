"""Walk-forward backtest: at each monthly rebalance date, rank and optimise
using only a rolling window of history strictly before that date, then hold
those weights (letting them drift with prices) until the next rebalance.
Never train on full history.

Weights are held fixed between rebalances rather than continuously
rebalanced — a documented simplification that matches how most textbook
backtests work.
"""

import pandas as pd

from src import config
from src.data import features
from src.models import ranker
from src.portfolio import optimise


def get_rebalance_dates(prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """The last trading day of each month between start and end, inclusive."""
    dates_in_range = prices.index[(prices.index >= start) & (prices.index <= end)]
    return list(dates_in_range.to_series().resample(config.REBALANCE_FREQUENCY).max().dropna())


def _turnover(old_weights: dict, new_weights: dict) -> float:
    """Total absolute change in weight, summed across every ticker held before or after."""
    tickers = set(old_weights) | set(new_weights)
    return sum(abs(new_weights.get(ticker, 0.0) - old_weights.get(ticker, 0.0)) for ticker in tickers)


def _period_returns(period_prices: pd.DataFrame, weights: dict) -> tuple[pd.Series, dict]:
    """Daily portfolio returns while holding fixed weights over one period.

    Also returns how those weights drifted by the end of the period as
    prices moved — that drifted position, not the original target weights,
    is what determines real turnover at the *next* rebalance.
    """
    entry_prices = period_prices[list(weights)]
    growth = entry_prices / entry_prices.iloc[0]

    weights_series = pd.Series(weights)
    portfolio_value = growth.dot(weights_series)
    daily_returns = portfolio_value.pct_change().fillna(0.0)

    end_value_per_ticker = weights_series * growth.iloc[-1]
    drifted_weights = (end_value_per_ticker / end_value_per_ticker.sum()).to_dict()
    return daily_returns, drifted_weights


def _select_target_weights(
    universe_prices: pd.DataFrame,
    dividends: pd.DataFrame,
    as_of_date: pd.Timestamp,
    rank_fn=ranker.rank_by_composite_score,
    optimiser_fn=optimise.max_sharpe_weights,
    risk_free_rate: pd.Series = None,
) -> dict[str, float]:
    """Rank the universe and optimise weights among the top holdings.

    rank_fn swaps in a different ranking rule (e.g. ranker.rank_by_momentum_only)
    and optimiser_fn swaps in a different portfolio construction method —
    neither changes anything else about the walk-forward mechanics.

    risk_free_rate is a date-indexed Series of actual historical rates (e.g.
    from FRED), or None to fall back to the flat config constant. Series.asof
    looks up the rate known as of as_of_date — never a later one, so this
    stays walk-forward-safe the same way every other feature does (rule 1).
    """
    scores = rank_fn(universe_prices, dividends, as_of_date)
    top_holdings = ranker.select_top_n(scores, n=config.TOP_N_HOLDINGS)

    training_window = features.training_window(universe_prices, as_of_date)
    rate = config.RISK_FREE_RATE_ANNUAL if risk_free_rate is None else risk_free_rate.asof(as_of_date)
    return optimiser_fn(training_window[list(top_holdings)], risk_free_rate=rate)


def _run_rebalanced_series(
    universe_prices: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    end: pd.Timestamp,
    get_target_weights,
) -> tuple[pd.Series, pd.Series, list[dict]]:
    """Shared walk-forward loop, used for both the strategy and the
    equal-weight benchmark: at each rebalance date, get target weights, pay
    the transaction cost for whatever turnover that requires, then hold
    until the next rebalance date.
    """
    gross_returns_by_period = []
    net_returns_by_period = []
    rebalance_log = []
    previous_weights: dict[str, float] = {}

    for i, rebalance_date in enumerate(rebalance_dates):
        is_last_period = i + 1 == len(rebalance_dates)
        period_end = end if is_last_period else rebalance_dates[i + 1]

        target_weights = get_target_weights(rebalance_date)
        turnover = _turnover(previous_weights, target_weights)
        cost = turnover * config.TRANSACTION_COST_BPS / 10_000

        period_prices = universe_prices.loc[
            (universe_prices.index >= rebalance_date) & (universe_prices.index <= period_end)
        ]
        if not is_last_period:
            period_prices = period_prices.iloc[:-1]  # next period's first day starts there instead

        gross_returns, previous_weights = _period_returns(period_prices, target_weights)
        net_returns = gross_returns.copy()
        net_returns.iloc[0] -= cost  # the transaction cost is a one-time hit on the rebalance day

        gross_returns_by_period.append(gross_returns)
        net_returns_by_period.append(net_returns)
        rebalance_log.append(
            {
                "date": rebalance_date,
                "selected_tickers": list(target_weights.keys()),
                "weights": target_weights,
                "turnover": turnover,
                "cost": cost,
            }
        )

    return pd.concat(gross_returns_by_period), pd.concat(net_returns_by_period), rebalance_log


def _strategy_returns(
    universe_prices: pd.DataFrame,
    dividends: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    end: pd.Timestamp,
    rank_fn=ranker.rank_by_composite_score,
    optimiser_fn=optimise.max_sharpe_weights,
    risk_free_rate: pd.Series = None,
) -> tuple[pd.Series, pd.Series, list[dict]]:
    def get_target_weights(rebalance_date: pd.Timestamp) -> dict[str, float]:
        return _select_target_weights(
            universe_prices, dividends, rebalance_date, rank_fn, optimiser_fn, risk_free_rate
        )

    return _run_rebalanced_series(universe_prices, rebalance_dates, end, get_target_weights)


def _equal_weight_returns(
    universe_prices: pd.DataFrame, rebalance_dates: list[pd.Timestamp], end: pd.Timestamp
) -> tuple[pd.Series, pd.Series, list[dict]]:
    tickers = list(universe_prices.columns)
    equal_weights = {ticker: 1 / len(tickers) for ticker in tickers}
    return _run_rebalanced_series(universe_prices, rebalance_dates, end, lambda _: equal_weights)


def _benchmark_returns(
    benchmark_prices: pd.Series, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.Series, pd.Series]:
    """SPY buy-and-hold: enter once at the first rebalance date, no further trading."""
    period_prices = benchmark_prices.loc[(benchmark_prices.index >= start) & (benchmark_prices.index <= end)]
    gross_returns = period_prices.pct_change().fillna(0.0)

    entry_cost = 1.0 * config.TRANSACTION_COST_BPS / 10_000  # buying from cash is 100% turnover
    net_returns = gross_returns.copy()
    net_returns.iloc[0] -= entry_cost
    return gross_returns, net_returns


def run_backtest(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    tickers: list[str] = config.TICKER_UNIVERSE,
    benchmark_ticker: str = config.BENCHMARK_TICKER,
    start: pd.Timestamp = config.BACKTEST_START_DATE,
    end: pd.Timestamp = config.BACKTEST_END_DATE,
    rank_fn=ranker.rank_by_composite_score,
    optimiser_fn=optimise.max_sharpe_weights,
    risk_free_rate: pd.Series = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward backtest: strategy vs. equal-weight and S&P 500
    benchmarks (rule 5), each reported gross and net of the
    configured transaction cost (rule 3).

    rank_fn selects the ranking rule (default: the momentum/low-vol/
    dividend-yield composite in ranker.rank_by_composite_score). Pass
    ranker.rank_by_momentum_only to compare a pure-momentum variant.
    optimiser_fn selects the portfolio construction method (default:
    mean-variance max Sharpe with Ledoit-Wolf shrinkage).
    risk_free_rate is a date-indexed Series of actual historical rates
    (e.g. from ingest.load_or_fetch_risk_free_rate), or None to fall back
    to the flat config.RISK_FREE_RATE_ANNUAL constant.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    universe_prices = prices[tickers]
    benchmark_prices = prices[benchmark_ticker]

    rebalance_dates = get_rebalance_dates(universe_prices, start, end)

    strategy_gross, strategy_net, rebalance_log = _strategy_returns(
        universe_prices, dividends, rebalance_dates, end, rank_fn, optimiser_fn, risk_free_rate
    )
    equal_weight_gross, equal_weight_net, _ = _equal_weight_returns(universe_prices, rebalance_dates, end)
    sp500_gross, sp500_net = _benchmark_returns(benchmark_prices, rebalance_dates[0], end)

    daily_returns = pd.DataFrame(
        {
            "strategy_gross": strategy_gross,
            "strategy_net": strategy_net,
            "equal_weight_gross": equal_weight_gross,
            "equal_weight_net": equal_weight_net,
            "sp500_gross": sp500_gross,
            "sp500_net": sp500_net,
        }
    )
    rebalance_log_df = pd.DataFrame(rebalance_log).set_index("date")
    return daily_returns, rebalance_log_df
