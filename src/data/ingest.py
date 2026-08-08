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
import time
from datetime import date

import pandas as pd
import requests
import yfinance as yf
from pandas_datareader import data as pdr
from yfinance.exceptions import YFRateLimitError

from src import config
from src.data import storage


class _BadDownloadError(Exception):
    """Raised when yf.download() returns without error but with data that
    looks like a failed fetch - confirmed directly that this happens for
    real: a batched call can silently come back with an entire ticker's
    column all-NaN, no exception raised at all, and that garbage would
    otherwise get cached as if it were good data.
    """


def _download_with_retry(tickers: list[str], start: date, end: date, is_valid=None, **kwargs) -> pd.DataFrame:
    """yf.download(), retrying with exponential backoff if Yahoo rate-limits
    the request, or if is_valid rejects a non-raising but bad response.

    Both prices and dividends are sourced through this one batched endpoint
    (see fetch_dividends) rather than yf.Ticker(...).dividends' separate
    per-ticker endpoint - verified directly that a sequential per-ticker
    dividend loop reliably tripped Yahoo's rate limit, on this machine and
    the deployed app alike, while this batched call has stayed reliable
    throughout. yf.download() already defaults to a 10s per-request timeout.
    """
    for attempt in range(config.YFINANCE_DOWNLOAD_MAX_RETRIES + 1):
        try:
            wide = yf.download(tickers, start=start, end=end, progress=False, **kwargs)
            if is_valid is not None and not is_valid(wide):
                raise _BadDownloadError()
            return wide
        except (YFRateLimitError, _BadDownloadError):
            if attempt == config.YFINANCE_DOWNLOAD_MAX_RETRIES:
                raise
            time.sleep(config.YFINANCE_DOWNLOAD_BACKOFF_BASE_SECONDS * (2**attempt))


def _has_no_all_nan_ticker(wide: pd.DataFrame) -> bool:
    """False if any requested ticker's Adj Close is entirely NaN.

    Every ticker in this project's universe traded during at least part of
    any requested range, so a wholly-NaN column is never a legitimate "no
    data" case (unlike dividends, where a ticker legitimately paying zero
    dividends is normal and expected) - it's the signature of yf.download
    silently failing for that ticker.
    """
    if wide.empty or "Adj Close" not in wide.columns.get_level_values(0):
        return False
    return not wide["Adj Close"].isna().all().any()


def fetch_price_history(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Daily OHLCV and adjusted close for every ticker.

    Columns: date, ticker, open, high, low, close, adj_close, volume.
    """
    wide = _download_with_retry(tickers, start, end, is_valid=_has_no_all_nan_ticker, auto_adjust=False)
    long = wide.stack(level="Ticker", future_stack=True).reset_index()
    long.columns = [str(column).lower().replace(" ", "_") for column in long.columns]
    return long.sort_values(["ticker", "date"]).reset_index(drop=True)


def fetch_dividends(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Real historical dividend payments: date, ticker, dividend_amount.

    Sourced from the same batched yf.download(..., actions=True) call as
    fetch_price_history, not yf.Ticker(...).dividends' separate per-ticker
    endpoint - see _download_with_retry's docstring for why.

    These are the actual amounts paid on their actual payment dates, not
    restated fundamentals — that's what keeps the dividend-yield factor
    (src/data/features.py) free of lookahead bias. yfinance pads
    non-payment days with 0.0 in this column, so those rows are dropped.
    """
    wide = _download_with_retry(tickers, start, end, actions=True, auto_adjust=False)
    long = wide["Dividends"].stack(future_stack=True).reset_index()
    long.columns = ["date", "ticker", "dividend_amount"]
    long = long[long["dividend_amount"] != 0]
    return long.sort_values(["ticker", "date"]).reset_index(drop=True)


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
