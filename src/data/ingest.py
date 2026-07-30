"""Pulls daily price and dividend history from yfinance.

Both functions return tidy "long" DataFrames (one row per date-ticker pair)
rather than yfinance's default wide MultiIndex — long format is trivial to
filter, inspect, and reason about column by column.
"""

import os
from datetime import date

import pandas as pd
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


def load_or_fetch_prices(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Read cached price history if present, otherwise fetch and cache it.

    Pass 1 has one fixed ticker universe and date range (config.py), so a
    single cache file per dataset is enough — it doesn't check whether the
    cached tickers/dates still match the request. Revisit this if the
    universe or date range starts changing between runs.
    """
    path = os.path.join(config.CACHE_DIR, "prices.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    prices = fetch_price_history(tickers, start, end)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    prices.to_parquet(path)
    return prices


def load_or_fetch_dividends(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Read cached dividend history if present, otherwise fetch and cache it."""
    path = os.path.join(config.CACHE_DIR, "dividends.parquet")
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
