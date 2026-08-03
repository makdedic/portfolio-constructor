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


@st.cache_data(show_spinner="Fetching sector classifications...")
def load_sp500_sectors_cached(tickers: tuple[str, ...]) -> dict[str, str]:
    return ingest.load_or_fetch_sp500_sectors(list(tickers))


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
daily_returns = results["daily_returns"]
rebalance_log = results["rebalance_log"]
st.caption(f"Showing: {st.session_state['config_caption']}")


def _total_and_cagr_pct(returns: pd.Series, n_years: float) -> tuple[float, float]:
    growth = (1 + returns).cumprod().iloc[-1]
    return (growth - 1) * 100, (growth ** (1 / n_years) - 1) * 100


n_years = (daily_returns.index[-1] - daily_returns.index[0]).days / 365.25
strategy_total, strategy_cagr = _total_and_cagr_pct(daily_returns["strategy_net"], n_years)
spy_total, spy_cagr = _total_and_cagr_pct(daily_returns["sp500_net"], n_years)
gross_growth = (1 + daily_returns["strategy_gross"]).cumprod().iloc[-1]
net_growth = (1 + daily_returns["strategy_net"]).cumprod().iloc[-1]
cost_drag_pct = (gross_growth - net_growth) * 100
avg_turnover_pct = rebalance_log["turnover"].mean() * 100

st.header("Headline performance")
kpi_cols = st.columns(4)
kpi_cols[0].metric("Total return", f"{strategy_total:.1f}%", f"{strategy_total - spy_total:+.1f}pp vs SPY")
kpi_cols[1].metric(
    "Annualised return (CAGR)", f"{strategy_cagr:.1f}%", f"{strategy_cagr - spy_cagr:+.1f}pp vs SPY"
)
kpi_cols[2].metric(
    "Cost drag",
    f"-{cost_drag_pct:.1f}pp",
    help="Total return given up to transaction costs — gross minus net total return.",
)
kpi_cols[3].metric(
    "Avg. turnover / rebalance",
    f"{avg_turnover_pct:.0f}%",
    help="Average share of the portfolio traded at each rebalance.",
)

rebalance_dates_newest_first = list(rebalance_log.index[::-1])
selected_rebalance_date = st.selectbox(
    "View holdings as of",
    options=rebalance_dates_newest_first,
    format_func=lambda d: d.date().isoformat(),
)
selected_weights = rebalance_log["weights"].loc[selected_rebalance_date]
weights_series = pd.Series(selected_weights).sort_values(ascending=False)
weights_series = weights_series[weights_series > 0]

st.header(f"Holdings as of {selected_rebalance_date.date()}")
weights_col, table_col = st.columns([2, 1])
with weights_col:
    st.bar_chart(weights_series)
with table_col:
    st.dataframe(weights_series.rename("weight").to_frame().round(4))

st.header("Most-held positions across the full backtest")
st.caption(
    "Sum of weight held at every rebalance — shows what the strategy consistently favours, "
    "not just its current picks."
)
weights_over_time = pd.DataFrame(list(rebalance_log["weights"]), index=rebalance_log.index).fillna(0.0)
top_holdings = weights_over_time.sum().sort_values(ascending=False).head(15)
persistent_col, persistent_table_col = st.columns([2, 1])
with persistent_col:
    st.bar_chart(top_holdings)
with persistent_table_col:
    st.dataframe(top_holdings.rename("total weight").to_frame().round(2))

st.header("Holdings count and sector composition over time")
st.caption(
    "Sector uses today's GICS classification applied across the whole backtest — a ticker "
    "reclassified between sectors historically wouldn't show that change, same simplification "
    "already used for today's S&P 500 constituent list."
)
holdings_count = rebalance_log["selected_tickers"].apply(len)
st.line_chart(holdings_count.rename("number of holdings"))

all_held_tickers = tuple(sorted(set().union(*rebalance_log["selected_tickers"])))
sector_lookup = load_sp500_sectors_cached(all_held_tickers)


def _sector_weights(weights: dict) -> dict:
    sector_totals: dict[str, float] = {}
    for ticker, weight in weights.items():
        sector = sector_lookup.get(ticker, "Unknown")
        sector_totals[sector] = sector_totals.get(sector, 0.0) + weight
    return sector_totals


sector_weights_over_time = (
    pd.DataFrame([_sector_weights(w) for w in rebalance_log["weights"]], index=rebalance_log.index).fillna(0.0)
    * 100
)
st.area_chart(sector_weights_over_time)

st.header("Risk metrics: strategy vs. benchmarks")
st.caption("All metrics computed net of the configured transaction cost.")
st.dataframe(results["risk_comparison"].round(4))

st.header("Walk-forward backtest: cumulative growth of $1 (net of costs)")
cumulative_growth = (1 + daily_returns[["strategy_net", "equal_weight_net", "sp500_net"]]).cumprod()
cumulative_growth.columns = ["Strategy", "Equal-weight", "SPY"]
st.line_chart(cumulative_growth)

st.header("Drawdown over time")
st.caption("Peak-to-trough decline from the running high — strategy vs. SPY.")


def _drawdown_pct(returns: pd.Series) -> pd.Series:
    cumulative_value = (1 + returns).cumprod()
    running_peak = cumulative_value.cummax()
    return (cumulative_value - running_peak) / running_peak * 100


drawdown_df = pd.DataFrame(
    {
        "Strategy": _drawdown_pct(daily_returns["strategy_net"]),
        "SPY": _drawdown_pct(daily_returns["sp500_net"]),
    }
)
st.area_chart(drawdown_df)
