"""Pulls daily price and dividend history from yfinance.

Both functions return tidy "long" DataFrames (one row per date-ticker pair)
rather than yfinance's default wide MultiIndex — long format is trivial to
filter, inspect, and reason about column by column.
"""

import io
import os
from datetime import date

import pandas as pd
import requests
import yfinance as yf
from pandas_datareader import data as pdr

from src import config


def fetch_price_history(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Daily OHLCV and adjusted close for every ticker.

    Columns: date, ticker, open, high, low, close, adj_close, volume.
    """
    wide = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    long = wide.stack(level="Ticker", future_stack=True).reset_index()
    long.columns = [str(column).lower().replace(" ", "_") for column in long.columns]
    return long.sort_values(["ticker", "date"]).reset_index(drop=True)


def fetch_dividends(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Real historical dividend payments: date, ticker, dividend_amount.

    These are the actual amounts paid on their actual payment dates, not
    restated fundamentals — that's what keeps the dividend-yield factor
    (src/data/features.py) free of lookahead bias.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    rows = []
    for ticker in tickers:
        dividends = yf.Ticker(ticker).dividends
        dividend_dates = dividends.index.tz_convert(None).normalize()
        for dividend_date, amount in zip(dividend_dates, dividends.to_numpy()):
            if start_ts <= dividend_date <= end_ts:
                rows.append({"date": dividend_date, "ticker": ticker, "dividend_amount": amount})
    return pd.DataFrame(rows, columns=["date", "ticker", "dividend_amount"]).sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)


def to_wide_adj_close(price_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot tidy prices to a date-indexed, ticker-columned adjusted close matrix.

    This is the shape factors, the optimiser, and the backtest all consume.
    """
    return price_long.pivot(index="date", columns="ticker", values="adj_close")


def load_or_fetch_prices(
    tickers: list[str], start: date, end: date, cache_name: str = "prices"
) -> pd.DataFrame:
    """Read cached price history if present, otherwise fetch and cache it.

    cache_name distinguishes runs against different universes (e.g. the dev
    ticker subset vs. the full S&P 500) so they don't clobber or misread
    each other's cache file — pass a distinct name whenever tickers isn't
    config.TICKER_UNIVERSE.
    """
    path = os.path.join(config.CACHE_DIR, f"{cache_name}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    prices = fetch_price_history(tickers, start, end)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    prices.to_parquet(path)
    return prices


def load_or_fetch_dividends(
    tickers: list[str], start: date, end: date, cache_name: str = "dividends"
) -> pd.DataFrame:
    """Read cached dividend history if present, otherwise fetch and cache it.

    cache_name distinguishes runs against different universes, same as
    load_or_fetch_prices.
    """
    path = os.path.join(config.CACHE_DIR, f"{cache_name}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    dividends = fetch_dividends(tickers, start, end)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    dividends.to_parquet(path)
    return dividends


def fetch_risk_free_rate(start: date, end: date) -> pd.DataFrame:
    """Daily annualised risk-free rate from FRED's 3-month T-bill series
    (config.RISK_FREE_RATE_FRED_SERIES): date, risk_free_rate_annual.

    FRED quotes this as a percentage and only publishes on days the
    Treasury reports — divide by 100 and forward-fill gaps so every
    trading day in range has a value.
    """
    raw = pdr.DataReader(config.RISK_FREE_RATE_FRED_SERIES, "fred", start, end)
    rate = (raw[config.RISK_FREE_RATE_FRED_SERIES] / 100).ffill().rename("risk_free_rate_annual")
    return rate.reset_index().rename(columns={"DATE": "date"})


def load_or_fetch_risk_free_rate(start: date, end: date) -> pd.DataFrame:
    """Read cached risk-free rate history if present, otherwise fetch and cache it."""
    path = os.path.join(config.CACHE_DIR, "risk_free_rate.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    rate = fetch_risk_free_rate(start, end)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    rate.to_parquet(path)
    return rate


def fetch_sp500_tickers() -> list[str]:
    """Current S&P 500 constituent tickers, scraped from Wikipedia.

    Wikipedia uses "." for share classes (e.g. "BRK.B"); yfinance expects
    "-" (e.g. "BRK-B") - normalised here. match="Symbol" targets the
    constituent table by its header rather than a positional index, since
    Wikipedia's table ordering on this page has changed before.

    A plain pd.read_html(url) call gets a 403 from Wikipedia - its default
    request has no User-Agent header, which Wikipedia blocks as bot-like -
    so the page is fetched directly first with one set.
    """
    response = requests.get(config.SP500_WIKIPEDIA_URL, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text), match="Symbol")
    tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False)
    return sorted(tickers.tolist())


def load_or_fetch_sp500_tickers() -> list[str]:
    """Read cached constituent list if present, otherwise fetch and cache it.

    Cached (not re-scraped every run) so a given backtest's universe stays
    stable and reproducible even if Wikipedia's page changes later.
    """
    path = os.path.join(config.CACHE_DIR, "sp500_tickers.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)["ticker"].tolist()
    tickers = fetch_sp500_tickers()
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_parquet(path)
    return tickers


def tickers_with_complete_history(wide_prices: pd.DataFrame) -> list[str]:
    """Tickers with no missing price anywhere in wide_prices' date range.

    Applied once, up front, when building a large/loosely-curated universe
    (the full S&P 500) rather than a hand-picked one: a recently-added
    constituent without this much history is excluded from the whole
    backtest, rather than requiring every downstream factor/ranking
    function to handle partial history individually.
    """
    return wide_prices.columns[wide_prices.notna().all()].tolist()
