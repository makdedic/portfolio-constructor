"""Builds the committed seed data under data/seed/ (see config.SEED_DIR):
a full S&P 500 snapshot (prices, dividends, risk-free rate, tickers/sectors)
that ships with the deployed app so it starts already warm, instead of
live-fetching from Yahoo/FRED on Streamlit Cloud's shared IP - confirmed
directly that IP gets rate-limited independent of our own request volume.

Run locally (where fetching is reliable) whenever the deployed data should
be refreshed:

    python scripts/build_seed.py

Only date, ticker, adj_close are kept for prices - open/high/low/close/
volume are never read anywhere in this codebase (verified via grep), so
seeding them would only add file size for no benefit. Parquet with ZSTD
compression, not the DuckDB file itself: the DuckDB file for this same data
is ~189MB, vs ~9MB as compressed Parquet - too heavy to commit otherwise.
"""

from src import config
from src.data import ingest

config.SEED_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    constituents = ingest.fetch_sp500_constituents()
    constituents.to_parquet(config.SEED_DIR / "sp500_tickers.parquet", compression="zstd")
    tickers = constituents["ticker"].tolist()
    print(f"tickers: {len(tickers)}")

    # config.BENCHMARK_TICKER (SPY) isn't itself an S&P 500 constituent - an
    # ETF that tracks the index, not one of the 500 companies in it - so it
    # has to be added explicitly or the benchmark has no price data at all.
    price_tickers = tickers + [config.BENCHMARK_TICKER]
    prices = ingest.fetch_price_history(price_tickers, config.DATA_START_DATE, config.DATA_END_DATE)
    prices[["date", "ticker", "adj_close"]].to_parquet(
        config.SEED_DIR / "prices.parquet", compression="zstd"
    )
    print(f"prices: {len(prices)} rows, through {prices['date'].max().date()}")

    dividends = ingest.fetch_dividends(tickers, config.DATA_START_DATE, config.DATA_END_DATE)
    dividends.to_parquet(config.SEED_DIR / "dividends.parquet", compression="zstd")
    print(f"dividends: {len(dividends)} rows, through {dividends['date'].max().date()}")

    rate = ingest.fetch_risk_free_rate(config.DATA_START_DATE, config.DATA_END_DATE)
    rate.to_parquet(config.SEED_DIR / "risk_free_rate.parquet", compression="zstd")
    print(f"risk_free_rate: {len(rate)} rows, through {rate['date'].max().date()}")


if __name__ == "__main__":
    main()
