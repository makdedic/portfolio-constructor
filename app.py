"""Streamlit dashboard: pick a ticker universe, run the walk-forward
backtest, see the current holdings, risk metrics, and backtest chart.
"""

import math

import pandas as pd
import streamlit as st

from src import config
from src.data import pipeline

st.set_page_config(page_title="Portfolio Constructor", layout="wide")
st.title("Portfolio Constructor")
st.caption(
    "Momentum + low-volatility + dividend-yield factor strategy, mean-variance "
    "optimised with Ledoit-Wolf shrinkage, walk-forward backtested against "
    "equal-weight and S&P 500 benchmarks."
)


@st.cache_data(show_spinner="Running pipeline (ingesting data and walk-forward backtesting)...")
def run_pipeline_cached(tickers: tuple[str, ...]) -> dict:
    return pipeline.run_pipeline(tickers=list(tickers))


st.sidebar.header("Ticker universe")
selected_tickers = st.sidebar.multiselect(
    "Stocks to include in the strategy universe",
    options=config.TICKER_UNIVERSE,
    default=config.TICKER_UNIVERSE,
)
run_clicked = st.sidebar.button("Run backtest", type="primary")

if not run_clicked:
    st.info("Choose a ticker universe in the sidebar and click **Run backtest** to get started.")
    st.stop()

# Fewer tickers than this makes MAX_WEIGHT_PER_STOCK infeasible: the
# optimiser can never make weights that are each capped at, say, 15% sum to
# 100% with fewer than ~7 stocks to spread them across.
min_tickers = math.ceil(1 / config.MAX_WEIGHT_PER_STOCK)
if len(selected_tickers) < min_tickers:
    st.error(
        f"Select at least {min_tickers} tickers — fewer makes the "
        f"{config.MAX_WEIGHT_PER_STOCK:.0%} per-stock weight cap infeasible."
    )
    st.stop()

results = run_pipeline_cached(tuple(selected_tickers))

latest_rebalance_date = results["rebalance_log"].index[-1]
latest_weights = results["rebalance_log"]["weights"].iloc[-1]
weights_series = pd.Series(latest_weights).sort_values(ascending=False)
weights_series = weights_series[weights_series > 0]

st.header(f"Current holdings (as of {latest_rebalance_date.date()})")
weights_col, table_col = st.columns([2, 1])
with weights_col:
    st.bar_chart(weights_series)
with table_col:
    st.dataframe(weights_series.rename("weight").to_frame().round(4))

st.header("Risk metrics: strategy vs. benchmarks")
st.caption("All metrics computed net of the configured transaction cost.")
st.dataframe(results["risk_comparison"].round(4))

st.header("Walk-forward backtest: cumulative growth of $1 (net of costs)")
cumulative_growth = (
    1 + results["daily_returns"][["strategy_net", "equal_weight_net", "sp500_net"]]
).cumprod()
cumulative_growth.columns = ["Strategy", "Equal-weight", "SPY"]
st.line_chart(cumulative_growth)
