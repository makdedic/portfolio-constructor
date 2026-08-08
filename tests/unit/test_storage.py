"""Guards the DuckDB storage layer's coverage semantics (tickers_needing_
fetch, log_fetch, upsert, query) independent of where data actually comes
from - ingest.py's integration with this layer (mocking the real network
calls) is tested separately in test_ingest.py.

Uses a tmp_path-based file, not ":memory:" - get_connection() opens fresh
per call by design (see its docstring), and DuckDB's in-memory databases
don't persist across separate connect() calls, so ":memory:" would
silently break any test that calls storage functions more than once.
"""

from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from src import config
from src.data import storage


@pytest.fixture
def conn(tmp_path):
    connection = storage.get_connection(tmp_path / "test.duckdb")
    yield connection
    connection.close()


def _prices_df(ticker: str, dates: list[date]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": [ticker] * len(dates),
            "open": [1.0] * len(dates),
            "high": [1.0] * len(dates),
            "low": [1.0] * len(dates),
            "close": [1.0] * len(dates),
            "adj_close": [1.0] * len(dates),
            "volume": [100.0] * len(dates),
        }
    )


def test_tickers_needing_fetch_returns_only_the_uncovered_subset(conn):
    storage.log_fetch(conn, "prices", ["AAPL"], date(2020, 1, 1), date(2020, 12, 31))

    missing = storage.tickers_needing_fetch(conn, "prices", ["AAPL", "MSFT"], date(2020, 1, 1), date(2020, 12, 31))

    assert missing == ["MSFT"]


def test_incremental_ticker_coverage_across_two_requests(conn):
    # First request: A and B are both missing.
    first_missing = storage.tickers_needing_fetch(conn, "prices", ["A", "B"], date(2020, 1, 1), date(2020, 1, 31))
    assert first_missing == ["A", "B"]
    storage.upsert_prices(
        conn, pd.concat([_prices_df("A", [date(2020, 1, 15)]), _prices_df("B", [date(2020, 1, 15)])])
    )
    storage.log_fetch(conn, "prices", ["A", "B"], date(2020, 1, 1), date(2020, 1, 31))

    # Second request, same range, one new ticker added: only C is missing.
    second_missing = storage.tickers_needing_fetch(
        conn, "prices", ["A", "B", "C"], date(2020, 1, 1), date(2020, 1, 31)
    )
    assert second_missing == ["C"]
    storage.upsert_prices(conn, _prices_df("C", [date(2020, 1, 15)]))
    storage.log_fetch(conn, "prices", ["C"], date(2020, 1, 1), date(2020, 1, 31))

    result = storage.query_prices(conn, ["A", "B", "C"], date(2020, 1, 1), date(2020, 1, 31))
    assert set(result["ticker"]) == {"A", "B", "C"}


def test_zero_row_result_still_counts_as_covered(conn):
    # A ticker that legitimately paid zero dividends: the upsert is empty,
    # but the fetch must still be logged as covered - otherwise it would
    # look "never fetched" and refetch forever.
    empty_dividends = pd.DataFrame(columns=["date", "ticker", "dividend_amount"])
    storage.upsert_dividends(conn, empty_dividends)
    storage.log_fetch(conn, "dividends", ["GOOGL"], date(2020, 1, 1), date(2020, 12, 31))

    missing = storage.tickers_needing_fetch(conn, "dividends", ["GOOGL"], date(2020, 1, 1), date(2020, 12, 31))
    assert missing == []


def test_duplicate_fetch_and_log_is_idempotent(conn):
    # Simulates a Prefect retry: the exact same fetch+log happens twice.
    prices = _prices_df("AAPL", [date(2020, 1, 1), date(2020, 1, 2)])
    storage.upsert_prices(conn, prices)
    storage.log_fetch(conn, "prices", ["AAPL"], date(2020, 1, 1), date(2020, 1, 2))

    storage.upsert_prices(conn, prices)
    storage.log_fetch(conn, "prices", ["AAPL"], date(2020, 1, 1), date(2020, 1, 2))

    assert conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM fetch_log").fetchone()[0] == 1


def test_narrower_logged_range_does_not_cover_a_wider_request(conn):
    storage.log_fetch(conn, "prices", ["AAPL"], date(2020, 6, 1), date(2020, 6, 30))

    missing = storage.tickers_needing_fetch(conn, "prices", ["AAPL"], date(2020, 1, 1), date(2020, 12, 31))

    assert missing == ["AAPL"]


def test_risk_free_rate_coverage_and_round_trip(conn):
    assert storage.risk_free_rate_covered(conn, date(2020, 1, 1), date(2020, 12, 31)) is False

    storage.log_fetch(conn, "risk_free_rate", [storage.NO_TICKER], date(2020, 1, 1), date(2020, 12, 31))
    assert storage.risk_free_rate_covered(conn, date(2020, 1, 1), date(2020, 12, 31)) is True

    rate = pd.DataFrame({"date": [date(2020, 1, 1)], "risk_free_rate_annual": [0.02]})
    storage.upsert_risk_free_rate(conn, rate)
    result = storage.query_risk_free_rate(conn, date(2020, 1, 1), date(2020, 12, 31))
    assert result["risk_free_rate_annual"].iloc[0] == pytest.approx(0.02)


def test_sp500_tickers_cache_hit_and_miss(conn):
    assert storage.sp500_tickers_cached(conn) == []

    constituents = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "sector": ["Information Technology"] * 2})
    storage.replace_sp500_constituents(conn, constituents)
    assert storage.sp500_tickers_cached(conn) == ["AAPL", "MSFT"]


def test_sp500_sectors_cached_excludes_unknown_tickers_rather_than_erroring(conn):
    constituents = pd.DataFrame(
        {"ticker": ["AAPL", "JPM"], "sector": ["Information Technology", "Financials"]}
    )
    storage.replace_sp500_constituents(conn, constituents)

    sectors = storage.sp500_sectors_cached(conn, ["AAPL", "JPM", "NOT_A_REAL_TICKER"])

    assert sectors == {"AAPL": "Information Technology", "JPM": "Financials"}


def test_replace_sp500_constituents_overwrites_the_previous_set(conn):
    storage.replace_sp500_constituents(
        conn, pd.DataFrame({"ticker": ["AAPL"], "sector": ["Information Technology"]})
    )
    storage.replace_sp500_constituents(
        conn, pd.DataFrame({"ticker": ["JPM"], "sector": ["Financials"]})
    )

    assert storage.sp500_tickers_cached(conn) == ["JPM"]


def test_get_connection_migrates_a_pre_existing_single_column_sp500_tickers_table(tmp_path):
    db_path = tmp_path / "old_schema.duckdb"

    # Simulate a cache file from before the sector column existed.
    import duckdb

    old_conn = duckdb.connect(str(db_path))
    old_conn.execute("CREATE TABLE sp500_tickers (ticker VARCHAR PRIMARY KEY)")
    old_conn.execute("INSERT INTO sp500_tickers VALUES ('AAPL'), ('MSFT')")
    old_conn.close()

    migrated_conn = storage.get_connection(db_path)

    assert storage.sp500_tickers_cached(migrated_conn) == ["AAPL", "MSFT"]
    assert storage.sp500_sectors_cached(migrated_conn, ["AAPL", "MSFT"]) == {}
    migrated_conn.close()


def test_get_connection_loads_the_committed_seed_into_a_fresh_database(tmp_path):
    """A freshly deployed app (no existing cache file) should start already
    warm from the committed seed snapshot (scripts/build_seed.py) rather
    than needing to fetch anything live - confirmed directly that Streamlit
    Cloud's shared IP gets rate-limited independent of our own request
    pattern, so this is what makes the deployed app reliable at all.
    """
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    pd.DataFrame({"ticker": ["AAPL"], "sector": ["Information Technology"]}).to_parquet(
        seed_dir / "sp500_tickers.parquet"
    )
    pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-02"]), "ticker": ["AAPL"], "adj_close": [100.0]}
    ).to_parquet(seed_dir / "prices.parquet")
    pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-15"]), "ticker": ["AAPL"], "dividend_amount": [0.2]}
    ).to_parquet(seed_dir / "dividends.parquet")
    pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-02"]), "risk_free_rate_annual": [0.015]}
    ).to_parquet(seed_dir / "risk_free_rate.parquet")

    db_path = tmp_path / "cache" / "portfolio.duckdb"
    with patch("src.data.storage.config.SEED_DIR", seed_dir), patch(
        "src.data.storage.config.DUCKDB_PATH", db_path
    ):
        conn = storage.get_connection()  # path=None -> default path -> seed loading fires

        assert storage.sp500_tickers_cached(conn) == ["AAPL"]
        prices = storage.query_prices(conn, ["AAPL"], date(2020, 1, 1), date(2020, 1, 31))
        assert prices["adj_close"].iloc[0] == 100.0
        assert len(storage.query_dividends(conn, ["AAPL"], date(2020, 1, 1), date(2020, 1, 31))) == 1
        assert storage.risk_free_rate_covered(conn, date(2020, 1, 1), date(2020, 1, 2)) is True
        # Seeded data counts as already covered - no refetch needed for it.
        assert storage.tickers_needing_fetch(conn, "prices", ["AAPL"], config.DATA_START_DATE, date(2020, 1, 2)) == []
        conn.close()


def test_get_connection_skips_seed_loading_when_no_seed_is_committed(tmp_path):
    """Local dev before scripts/build_seed.py has ever been run - loading a
    non-existent seed must be a silent no-op, not an error.
    """
    db_path = tmp_path / "cache" / "portfolio.duckdb"
    with patch("src.data.storage.config.SEED_DIR", tmp_path / "no_seed_here"), patch(
        "src.data.storage.config.DUCKDB_PATH", db_path
    ):
        conn = storage.get_connection()
        assert storage.sp500_tickers_cached(conn) == []
        conn.close()
