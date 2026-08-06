"""Pulls daily price and dividend history from yfinance (plus FRED and a
Wikipedia scrape for the risk-free rate and S&P 500 constituents).

fetch_X functions hit the external source directly, with no caching
awareness. load_or_fetch_X functions cache via src.data.storage (DuckDB) —
only fetching tickers storage doesn't already have covered for the
requested range, so overlapping requests (e.g. the dev ticker subset and
the full S&P 500) share cached data instead of needing separate caches.

Price/dividend functions return tidy "long" DataFrames (one row per
date-ticker pair) rather than yfinance's default wide MultiIndex — long
format is trivial to filter, inspect, and reason about column by column.
"""

import io
from datetime import date

import pandas as pd
import requests
import yfinance as yf
from pandas_datareader import data as pdr

from src import config
from src.data import storage


class _TimeoutSession(requests.Session):
    """A requests.Session that enforces a default timeout on every call.

    yf.Ticker(...) accepts a session but no timeout - without this, a single
    hung request in fetch_dividends' sequential per-ticker loop could block
    it indefinitely, with no way to recover short of restarting the process.
    """

    def request(self, *args, **kwargs):
        kwargs.setdefault("timeout", config.YFINANCE_REQUEST_TIMEOUT_SECONDS)
        return super().request(*args, **kwargs)


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
    session = _TimeoutSession()
    rows = []
    for ticker in tickers:
        dividends = yf.Ticker(ticker, session=session).dividends
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
    """Read cached price history for tickers not already covered in storage
    for [start, end], fetch only what's missing, then return the full
    requested set from storage.
    """
    conn = storage.get_connection()
    missing = storage.tickers_needing_fetch(conn, "prices", tickers, start, end)
    if missing:
        fetched = fetch_price_history(missing, start, end)
        storage.upsert_prices(conn, fetched)
        storage.log_fetch(conn, "prices", missing, start, end)
    return storage.query_prices(conn, tickers, start, end)


def load_or_fetch_dividends(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Read cached dividend history for tickers not already covered in
    storage for [start, end], fetch only what's missing, then return the
    full requested set from storage.
    """
    conn = storage.get_connection()
    missing = storage.tickers_needing_fetch(conn, "dividends", tickers, start, end)
    if missing:
        fetched = fetch_dividends(missing, start, end)
        storage.upsert_dividends(conn, fetched)
        storage.log_fetch(conn, "dividends", missing, start, end)
    return storage.query_dividends(conn, tickers, start, end)


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
    """Read the cached risk-free rate for [start, end] if already covered,
    otherwise fetch and cache it.
    """
    conn = storage.get_connection()
    if not storage.risk_free_rate_covered(conn, start, end):
        rate = fetch_risk_free_rate(start, end)
        storage.upsert_risk_free_rate(conn, rate)
        storage.log_fetch(conn, "risk_free_rate", [storage.NO_TICKER], start, end)
    return storage.query_risk_free_rate(conn, start, end)


def fetch_sp500_constituents() -> pd.DataFrame:
    """Current S&P 500 constituents, scraped from Wikipedia: ticker, sector
    (GICS Sector).

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
    table = tables[0][["Symbol", "GICS Sector"]].rename(columns={"Symbol": "ticker", "GICS Sector": "sector"})
    table["ticker"] = table["ticker"].str.replace(".", "-", regex=False)
    return table.sort_values("ticker").reset_index(drop=True)


def fetch_sp500_tickers() -> list[str]:
    """Just the ticker symbols, for callers that don't need sector info."""
    return fetch_sp500_constituents()["ticker"].tolist()


def load_or_fetch_sp500_tickers() -> list[str]:
    """Read cached constituent list if present, otherwise fetch and cache it.

    Cached (not re-scraped every run) so a given backtest's universe stays
    stable and reproducible even if Wikipedia's page changes later.
    """
    conn = storage.get_connection()
    cached = storage.sp500_tickers_cached(conn)
    if cached:
        return cached
    constituents = fetch_sp500_constituents()
    storage.replace_sp500_constituents(conn, constituents)
    return constituents["ticker"].tolist()


def load_or_fetch_sp500_sectors(tickers: list[str]) -> dict[str, str]:
    """GICS sector for each of tickers, from the same cached S&P 500 scrape
    load_or_fetch_sp500_tickers uses - covers the dev universe too, since
    every DEV_TICKERS symbol is itself a current S&P 500 constituent
    (verified directly, not assumed).

    Uses today's sector classification across the whole backtest history -
    a ticker reclassified between sectors at some point wouldn't show that
    change. Same simplification already accepted for using today's
    constituent list (survivorship bias).
    """
    conn = storage.get_connection()
    sectors = storage.sp500_sectors_cached(conn, tickers)
    if len(sectors) < len(set(tickers)):
        constituents = fetch_sp500_constituents()
        storage.replace_sp500_constituents(conn, constituents)
        sectors = storage.sp500_sectors_cached(conn, tickers)
    return sectors
