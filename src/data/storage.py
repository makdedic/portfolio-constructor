"""DuckDB-backed storage for ingested data: the database connection,
schema, and per-ticker/date-range incremental caching logic.

ingest.py's fetch_X functions (the ones that actually hit yfinance/FRED/
Wikipedia) stay pure "hit the external source" calls with no caching
awareness — its load_or_fetch_X functions call into this module instead of
doing raw file I/O themselves, the same separation this codebase already
uses for Prefect (pipeline.py) vs. business logic.
"""

from pathlib import Path

import duckdb
import pandas as pd

from src import config

# Sentinel ticker for datasets with no ticker dimension (risk_free_rate) -
# fetch_log's schema always has a ticker column, so a single time series
# needs a stand-in value rather than a second, parallel coverage mechanism.
NO_TICKER = "__NONE__"


def get_connection(path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Opens (creating if needed) the DuckDB database and ensures every
    table this project uses exists.

    Opened fresh per call, never held module-level: DuckDB allows only one
    read-write connection per file across processes, so a short-lived
    connection minimises the window another process (the Streamlit app, a
    notebook) could collide with it. This doesn't eliminate that risk
    entirely - it's a real, accepted limitation, not solved here.
    """
    conn = duckdb.connect(str(path or config.DUCKDB_PATH))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS prices (
            date DATE, ticker VARCHAR, open DOUBLE, high DOUBLE, low DOUBLE,
            close DOUBLE, adj_close DOUBLE, volume DOUBLE, PRIMARY KEY (date, ticker)
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS dividends (
            date DATE, ticker VARCHAR, dividend_amount DOUBLE, PRIMARY KEY (date, ticker)
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS risk_free_rate (
            date DATE PRIMARY KEY, risk_free_rate_annual DOUBLE
        )"""
    )
    conn.execute("CREATE TABLE IF NOT EXISTS sp500_tickers (ticker VARCHAR PRIMARY KEY)")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS fetch_log (
            dataset VARCHAR, ticker VARCHAR, start_date DATE, end_date DATE,
            fetched_at TIMESTAMP, PRIMARY KEY (dataset, ticker, start_date, end_date)
        )"""
    )
    return conn


def tickers_needing_fetch(
    conn: duckdb.DuckDBPyConnection, dataset: str, tickers: list[str], start, end
) -> list[str]:
    """Which of tickers do NOT already have a logged fetch covering [start, end].

    Checks fetch_log, not the data table itself: a ticker can legitimately
    have zero rows in the data table after a complete, successful fetch
    (e.g. a stock that has never paid a dividend) - row-presence alone
    would treat that as "never fetched" and refetch it forever.
    """
    placeholders = ",".join(["?"] * len(tickers))
    covered = conn.execute(
        f"""SELECT DISTINCT ticker FROM fetch_log
            WHERE dataset = ? AND start_date <= ? AND end_date >= ? AND ticker IN ({placeholders})""",
        [dataset, start, end, *tickers],
    ).fetchall()
    covered_tickers = {row[0] for row in covered}
    return [ticker for ticker in tickers if ticker not in covered_tickers]


def log_fetch(conn: duckdb.DuckDBPyConnection, dataset: str, tickers: list[str], start, end) -> None:
    """Records that (dataset, ticker) was fetched for [start, end], regardless
    of whether the fetch produced any data rows.

    INSERT OR REPLACE, not a plain INSERT: Prefect's fetch tasks (pipeline.py)
    have retries=2, so the same ticker/range can genuinely be logged twice on
    a transient retry - this keeps that idempotent instead of erroring.
    """
    now = pd.Timestamp.now()
    conn.executemany(
        "INSERT OR REPLACE INTO fetch_log VALUES (?, ?, ?, ?, ?)",
        [(dataset, ticker, start, end, now) for ticker in tickers],
    )


def upsert_prices(conn: duckdb.DuckDBPyConnection, prices_long: pd.DataFrame) -> None:
    conn.register("_fetched_prices", prices_long)
    conn.execute(
        """INSERT OR REPLACE INTO prices (date, ticker, open, high, low, close, adj_close, volume)
           SELECT date, ticker, open, high, low, close, adj_close, volume FROM _fetched_prices"""
    )
    conn.unregister("_fetched_prices")


def query_prices(conn: duckdb.DuckDBPyConnection, tickers: list[str], start, end) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(tickers))
    return conn.execute(
        f"""SELECT date, ticker, open, high, low, close, adj_close, volume FROM prices
            WHERE ticker IN ({placeholders}) AND date BETWEEN ? AND ? ORDER BY ticker, date""",
        [*tickers, start, end],
    ).fetchdf()


def upsert_dividends(conn: duckdb.DuckDBPyConnection, dividends: pd.DataFrame) -> None:
    conn.register("_fetched_dividends", dividends)
    conn.execute(
        """INSERT OR REPLACE INTO dividends (date, ticker, dividend_amount)
           SELECT date, ticker, dividend_amount FROM _fetched_dividends"""
    )
    conn.unregister("_fetched_dividends")


def query_dividends(conn: duckdb.DuckDBPyConnection, tickers: list[str], start, end) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(tickers))
    return conn.execute(
        f"""SELECT date, ticker, dividend_amount FROM dividends
            WHERE ticker IN ({placeholders}) AND date BETWEEN ? AND ? ORDER BY ticker, date""",
        [*tickers, start, end],
    ).fetchdf()


def risk_free_rate_covered(conn: duckdb.DuckDBPyConnection, start, end) -> bool:
    """Whether a single logged fetch already covers [start, end] - no ticker
    dimension, so this checks NO_TICKER's coverage rather than a per-ticker set.
    """
    return bool(
        conn.execute(
            """SELECT 1 FROM fetch_log
               WHERE dataset = 'risk_free_rate' AND ticker = ? AND start_date <= ? AND end_date >= ?""",
            [NO_TICKER, start, end],
        ).fetchone()
    )


def upsert_risk_free_rate(conn: duckdb.DuckDBPyConnection, rate: pd.DataFrame) -> None:
    conn.register("_fetched_rate", rate)
    conn.execute(
        """INSERT OR REPLACE INTO risk_free_rate (date, risk_free_rate_annual)
           SELECT date, risk_free_rate_annual FROM _fetched_rate"""
    )
    conn.unregister("_fetched_rate")


def query_risk_free_rate(conn: duckdb.DuckDBPyConnection, start, end) -> pd.DataFrame:
    return conn.execute(
        "SELECT date, risk_free_rate_annual FROM risk_free_rate WHERE date BETWEEN ? AND ? ORDER BY date",
        [start, end],
    ).fetchdf()


def sp500_tickers_cached(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """The cached constituent list, or an empty list if never fetched -
    wholesale refresh-if-empty, same as the flat-file version this replaces.
    """
    rows = conn.execute("SELECT ticker FROM sp500_tickers ORDER BY ticker").fetchall()
    return [row[0] for row in rows]


def replace_sp500_tickers(conn: duckdb.DuckDBPyConnection, tickers: list[str]) -> None:
    conn.execute("DELETE FROM sp500_tickers")
    conn.executemany("INSERT INTO sp500_tickers VALUES (?)", [(ticker,) for ticker in tickers])
