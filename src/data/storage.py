"""DuckDB-backed storage for ingested data: the database connection,
schema, and per-ticker/date-range incremental caching logic.

ingest.py's fetch_X functions (the ones that actually hit yfinance/FRED/
Wikipedia) stay pure "hit the external source" calls with no caching
awareness — its load_or_fetch_X functions call into this module instead of
doing raw file I/O themselves.
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
    target = path or config.DUCKDB_PATH
    is_fresh = path is None and not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(target))
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
    conn.execute("CREATE TABLE IF NOT EXISTS sp500_tickers (ticker VARCHAR PRIMARY KEY, sector VARCHAR)")
    # Migrates any pre-existing cache file from before the sector column
    # existed - a no-op on an already-current or freshly created table.
    conn.execute("ALTER TABLE sp500_tickers ADD COLUMN IF NOT EXISTS sector VARCHAR")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS fetch_log (
            dataset VARCHAR, ticker VARCHAR, start_date DATE, end_date DATE,
            fetched_at TIMESTAMP, PRIMARY KEY (dataset, ticker, start_date, end_date)
        )"""
    )

    if is_fresh:
        _load_seed(conn)
    return conn


def _load_seed(conn: duckdb.DuckDBPyConnection) -> None:
    """Populates a freshly created default-path database from the committed
    seed snapshot (data/seed/, built by scripts/build_seed.py), if present -
    a no-op for local dev before that script has ever been run.

    Ships the deployed app already warm with a full S&P 500 dataset instead
    of live-fetching from Yahoo/FRED on first load - confirmed directly that
    Streamlit Cloud's shared IP gets rate-limited independent of our own
    request pattern, so no amount of client-side politeness fixes that.

    Only date, ticker, adj_close are seeded for prices (see
    scripts/build_seed.py for why) - open/high/low/close/volume stay NULL
    for these rows, which is safe since nothing in this codebase reads them.
    """
    tickers_path = config.SEED_DIR / "sp500_tickers.parquet"
    if not tickers_path.exists():
        return

    conn.execute(
        f"""INSERT INTO sp500_tickers (ticker, sector)
            SELECT ticker, sector FROM read_parquet('{tickers_path}')"""
    )
    conn.execute(
        f"""INSERT INTO prices (date, ticker, adj_close)
            SELECT date, ticker, adj_close FROM read_parquet('{config.SEED_DIR / "prices.parquet"}')"""
    )
    conn.execute(
        f"""INSERT INTO dividends (date, ticker, dividend_amount)
            SELECT date, ticker, dividend_amount FROM read_parquet('{config.SEED_DIR / "dividends.parquet"}')"""
    )
    conn.execute(
        f"""INSERT INTO risk_free_rate (date, risk_free_rate_annual)
            SELECT date, risk_free_rate_annual
            FROM read_parquet('{config.SEED_DIR / "risk_free_rate.parquet"}')"""
    )

    seeded_tickers = [row[0] for row in conn.execute("SELECT ticker FROM sp500_tickers").fetchall()]
    prices_through = conn.execute("SELECT MAX(date) FROM prices").fetchone()[0]
    dividends_through = conn.execute("SELECT MAX(date) FROM dividends").fetchone()[0]
    rate_through = conn.execute("SELECT MAX(date) FROM risk_free_rate").fetchone()[0]

    log_fetch(conn, "prices", seeded_tickers, config.DATA_START_DATE, prices_through)
    log_fetch(conn, "dividends", seeded_tickers, config.DATA_START_DATE, dividends_through)
    log_fetch(conn, "risk_free_rate", [NO_TICKER], config.DATA_START_DATE, rate_through)


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

    INSERT OR REPLACE, not a plain INSERT: the same ticker/range can
    genuinely get logged twice (a retried fetch, or two callers racing on
    the same uncached range) - this keeps that idempotent instead of
    erroring.
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


def sp500_sectors_cached(conn: duckdb.DuckDBPyConnection, tickers: list[str]) -> dict[str, str]:
    """Sector for whichever of tickers are already cached with one -
    tickers with no cached sector (or not cached at all) are simply absent
    from the result, not an error.
    """
    placeholders = ",".join(["?"] * len(tickers))
    rows = conn.execute(
        f"SELECT ticker, sector FROM sp500_tickers WHERE ticker IN ({placeholders}) AND sector IS NOT NULL",
        tickers,
    ).fetchall()
    return dict(rows)


def replace_sp500_constituents(conn: duckdb.DuckDBPyConnection, constituents: pd.DataFrame) -> None:
    """constituents: a ticker, sector DataFrame (see ingest.fetch_sp500_constituents)."""
    conn.execute("DELETE FROM sp500_tickers")
    conn.register("_constituents", constituents)
    conn.execute("INSERT INTO sp500_tickers (ticker, sector) SELECT ticker, sector FROM _constituents")
    conn.unregister("_constituents")
