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


def _training_window(universe_prices: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """The trailing TRAIN_WINDOW_YEARS of price history, strictly before as_of_date.

    This one slice is what enforces both rule 2 (rolling window, never full
    history) and, downstream, rule 1 (no lookahead) for everything computed
    from it.
    """
    window_start = as_of_date - pd.DateOffset(years=config.TRAIN_WINDOW_YEARS)
    return universe_prices.loc[(universe_prices.index >= window_start) & (universe_prices.index < as_of_date)]


def _select_target_weights(
    universe_prices: pd.DataFrame,
    dividends: pd.DataFrame,
    as_of_date: pd.Timestamp,
    score_fn=ranker.score_stocks,
) -> dict[str, float]:
    """Rank the universe and optimise weights using only the trailing training window.

    score_fn swaps in a different ranking rule (e.g. ranker.score_by_momentum_only)
    without changing anything else about the walk-forward mechanics.
    """
    training_window = _training_window(universe_prices, as_of_date)

    momentum = features.compute_momentum(training_window, as_of_date)
    low_volatility = features.compute_low_volatility(training_window, as_of_date)
    dividend_yield = features.compute_dividend_yield(dividends, training_window, as_of_date)

    scores = score_fn(momentum, low_volatility, dividend_yield)
    top_holdings = ranker.select_top_n(scores, n=config.TOP_N_HOLDINGS)

    return optimise.max_sharpe_weights(training_window[list(top_holdings)])


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
    score_fn=ranker.score_stocks,
) -> tuple[pd.Series, pd.Series, list[dict]]:
    def get_target_weights(rebalance_date: pd.Timestamp) -> dict[str, float]:
        return _select_target_weights(universe_prices, dividends, rebalance_date, score_fn)

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
    score_fn=ranker.score_stocks,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward backtest: strategy vs. equal-weight and S&P 500
    benchmarks (claude.md rule 5), each reported gross and net of the
    configured transaction cost (rule 3).

    score_fn selects the ranking rule (default: the momentum/low-vol/
    dividend-yield composite in ranker.score_stocks). Pass
    ranker.score_by_momentum_only to compare a pure-momentum variant.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    universe_prices = prices[tickers]
    benchmark_prices = prices[benchmark_ticker]

    rebalance_dates = get_rebalance_dates(universe_prices, start, end)

    strategy_gross, strategy_net, rebalance_log = _strategy_returns(
        universe_prices, dividends, rebalance_dates, end, score_fn
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
