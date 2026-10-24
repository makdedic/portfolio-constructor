"""Builds the committed seed data under data/seed/ (see config.SEED_DIR): a
point-in-time S&P 500 snapshot (prices, dividends, risk-free rate, tickers/
sectors, historical changes log) that ships with the deployed app so it
starts already warm, instead of live-fetching from Yahoo/FRED on Streamlit
Cloud's shared IP - confirmed directly that IP gets rate-limited
independent of our own request volume.

Run locally (where fetching is reliable) whenever the deployed data should
be refreshed:

    python scripts/build_seed.py

Only date, ticker, adj_close are kept for prices - open/high/low/close/
volume are never read anywhere in this codebase (verified via grep), so
seeding them would only add file size for no benefit. Parquet with ZSTD
compression, not the DuckDB file itself: the DuckDB file for this same data
is ~189MB, vs ~9MB as compressed Parquet - too heavy to commit otherwise.
"""

import pandas as pd

from src import config
from src.data import ingest

config.SEED_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    constituents = ingest.fetch_sp500_constituents()
    constituents.to_parquet(config.SEED_DIR / "sp500_tickers.parquet", compression="zstd")
    print(f"tickers: {len(constituents)}")

    changes = ingest.fetch_sp500_historical_changes()
    changes.to_parquet(config.SEED_DIR / "sp500_historical_changes.parquet", compression="zstd")
    print(f"historical changes: {len(changes)} rows")

    # Seed the point-in-time universe (reduces survivorship bias), not just
    # today's constituents - see ingest.reconstruct_sp500_membership. Mirrors
    # load_or_fetch_sp500_tickers's own current | tradeable(recovered) shape,
    # so the seed actually covers what that function returns at runtime -
    # historical_tickers alone omits any current constituent added after
    # as_of_date (recent index joins), which load_or_fetch_sp500_tickers
    # still includes.
    historical_tickers = ingest.reconstruct_sp500_membership(
        constituents["ticker"].tolist(), changes, config.BACKTEST_START_DATE
    )
    recovered = sorted(set(historical_tickers) - set(constituents["ticker"]))
    print(f"reconstructed as of {config.BACKTEST_START_DATE}: {len(recovered)} tickers recovered vs today's list")

    # config.BENCHMARK_TICKER (SPY) isn't itself an S&P 500 constituent - an
    # ETF that tracks the index, not one of the 500 companies in it - so it
    # has to be added explicitly or the benchmark has no price data at all.
    # validate=False: some reconstructed tickers are no longer tradeable at
    # all (delisted/acquired) - a real, expected outcome here, not a sign
    # of a bad fetch.
    price_tickers = sorted(set(constituents["ticker"]) | set(historical_tickers) | {config.BENCHMARK_TICKER})
    prices = ingest.fetch_price_history(
        price_tickers, config.DATA_START_DATE, config.DATA_END_DATE, validate=False
    )
    tradeable_tickers = sorted(prices.dropna(subset=["adj_close"])["ticker"].unique())
    prices = prices[prices["ticker"].isin(tradeable_tickers)]
    prices[["date", "ticker", "adj_close"]].to_parquet(
        config.SEED_DIR / "prices.parquet", compression="zstd"
    )
    print(f"prices: {len(prices)} rows ({len(tradeable_tickers)} tradeable), through {prices['date'].max().date()}")

    # Recorded so storage._load_seed can mark exactly these tickers as
    # already fetched - sp500_tickers alone only ever holds today's
    # constituents, not the wider reconstructed set actually seeded here.
    pd.DataFrame({"ticker": tradeable_tickers}).to_parquet(
        config.SEED_DIR / "seeded_tickers.parquet", compression="zstd"
    )

    dividend_tickers = sorted(set(tradeable_tickers) - {config.BENCHMARK_TICKER})
    dividends = ingest.fetch_dividends(dividend_tickers, config.DATA_START_DATE, config.DATA_END_DATE)
    dividends.to_parquet(config.SEED_DIR / "dividends.parquet", compression="zstd")
    print(f"dividends: {len(dividends)} rows, through {dividends['date'].max().date()}")

    rate = ingest.fetch_risk_free_rate(config.DATA_START_DATE, config.DATA_END_DATE)
    rate.to_parquet(config.SEED_DIR / "risk_free_rate.parquet", compression="zstd")
    print(f"risk_free_rate: {len(rate)} rows, through {rate['date'].max().date()}")


if __name__ == "__main__":
    main()
