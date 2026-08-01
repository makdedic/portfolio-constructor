"""Streamlit dashboard: pick a ranker, optimiser, ticker universe, and
risk-free rate, run the walk-forward backtest, see the current holdings,
risk metrics, and backtest chart.
"""

import math

import pandas as pd
import streamlit as st

from src import config
from src.data import ingest, pipeline
from src.models import lgbm_ranker, ranker
from src.portfolio import optimise

st.set_page_config(page_title="Portfolio Constructor", layout="wide")
st.title("Portfolio Constructor")
st.caption(
    "Pick a ranker, optimiser, ticker universe, and risk-free rate in the sidebar, then "
    "run a walk-forward backtest against equal-weight and S&P 500 benchmarks."
)

RANK_FN_OPTIONS = {
    "Composite (momentum + low-vol + dividend yield)": ranker.rank_by_composite_score,
    "Momentum only": ranker.rank_by_momentum_only,
    "LightGBM (ML-predicted next-month return)": lgbm_ranker.rank_by_lgbm,
}
OPTIMISER_FN_OPTIONS = {
    "Max-Sharpe (mean-variance, Ledoit-Wolf)": optimise.max_sharpe_weights,
    "Risk-parity (equal risk contribution)": optimise.risk_parity_weights,
}
DEV_UNIVERSE_LABEL = "Dev universe (~39 tickers, fast)"
SP500_UNIVERSE_LABEL = "Full S&P 500 (~500 tickers, slower first run)"


@st.cache_data(show_spinner="Running pipeline (ingesting data and walk-forward backtesting)...")
def run_pipeline_cached(
    tickers: tuple[str, ...], rank_choice: str, optimiser_choice: str, use_real_rate: bool
) -> dict:
    return pipeline.run_pipeline(
        tickers=list(tickers),
        rank_fn=RANK_FN_OPTIONS[rank_choice],
        optimiser_fn=OPTIMISER_FN_OPTIONS[optimiser_choice],
        use_real_risk_free_rate=use_real_rate,
    )


@st.cache_data(show_spinner="Fetching S&P 500 constituent list...")
def load_sp500_tickers_cached() -> list[str]:
    return ingest.load_or_fetch_sp500_tickers()


st.sidebar.header("Ticker universe")
universe_choice = st.sidebar.radio("Universe", [DEV_UNIVERSE_LABEL, SP500_UNIVERSE_LABEL])

if universe_choice == DEV_UNIVERSE_LABEL:
    selected_tickers = st.sidebar.multiselect(
        "Stocks to include in the strategy universe",
        options=config.TICKER_UNIVERSE,
        default=config.TICKER_UNIVERSE,
    )
else:
    st.sidebar.caption(
        "First run fetches and caches ~500 tickers (~1-2 min); subsequent runs are fast. "
        "Tickers without enough price history for the backtest range are excluded automatically."
    )
    selected_tickers = load_sp500_tickers_cached()

st.sidebar.header("Strategy")
rank_choice = st.sidebar.selectbox("Ranker", list(RANK_FN_OPTIONS.keys()))
optimiser_choice = st.sidebar.selectbox("Optimiser", list(OPTIMISER_FN_OPTIONS.keys()))

st.sidebar.header("Risk-free rate")
use_real_rate = st.sidebar.checkbox("Use real FRED risk-free rate (instead of flat 2%)")

run_clicked = st.sidebar.button("Run backtest", type="primary")

# st.button() only returns True on the exact rerun triggered by that click —
# every other widget on the page (like the date selector below) also
# triggers a full script rerun, on which the button reads False again. So
# results are stashed in session_state on click and reused on every rerun
# after that; otherwise changing the date selector would silently wipe the
# page and demand another click of "Run backtest" to see anything.
if run_clicked:
    if universe_choice == DEV_UNIVERSE_LABEL:
        # Fewer tickers than this makes MAX_WEIGHT_PER_STOCK infeasible: the
        # optimiser can never make weights that are each capped at, say, 15%
        # sum to 100% with fewer than ~7 stocks to spread them across. The
        # full S&P 500 universe is always well above this floor.
        min_tickers = math.ceil(1 / config.MAX_WEIGHT_PER_STOCK)
        if len(selected_tickers) < min_tickers:
            st.error(
                f"Select at least {min_tickers} tickers — fewer makes the "
                f"{config.MAX_WEIGHT_PER_STOCK:.0%} per-stock weight cap infeasible."
            )
            st.stop()
    st.session_state["results"] = run_pipeline_cached(
        tuple(selected_tickers), rank_choice, optimiser_choice, use_real_rate
    )
    rate_label = "real FRED risk-free rate" if use_real_rate else "flat 2% risk-free rate"
    st.session_state["config_caption"] = f"{rank_choice} · {optimiser_choice} · {universe_choice} · {rate_label}"

if "results" not in st.session_state:
    st.info("Choose a configuration in the sidebar and click **Run backtest** to get started.")
    st.stop()

results = st.session_state["results"]
st.caption(f"Showing: {st.session_state['config_caption']}")

rebalance_dates_newest_first = list(results["rebalance_log"].index[::-1])
selected_rebalance_date = st.selectbox(
    "View holdings as of",
    options=rebalance_dates_newest_first,
    format_func=lambda d: d.date().isoformat(),
)
selected_weights = results["rebalance_log"]["weights"].loc[selected_rebalance_date]
weights_series = pd.Series(selected_weights).sort_values(ascending=False)
weights_series = weights_series[weights_series > 0]

st.header(f"Holdings as of {selected_rebalance_date.date()}")
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
