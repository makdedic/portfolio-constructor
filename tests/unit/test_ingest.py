"""Guards the S&P 500 constituent sourcing: ticker normalisation for
yfinance compatibility, and that load_or_fetch_X correctly delegates
caching to src.data.storage (DuckDB) rather than duplicating that logic.
"""

from unittest.mock import patch

import pandas as pd

from src.data import ingest, storage


def _fake_wikipedia_response(tickers: list[str]) -> str:
    """A minimal HTML table shaped like Wikipedia's real constituent table -
    just enough for pd.read_html(..., match="Symbol") to find it.
    """
    table = pd.DataFrame({"Symbol": tickers, "Security": [f"Company {t}" for t in tickers]})
    return table.to_html(index=False)


def test_fetch_sp500_tickers_normalises_share_classes_for_yfinance():
    html = _fake_wikipedia_response(["AAPL", "BRK.B", "BF.B", "MSFT"])

    class _FakeResponse:
        text = html

        def raise_for_status(self):
            pass

    with patch("src.data.ingest.requests.get", return_value=_FakeResponse()):
        tickers = ingest.fetch_sp500_tickers()

    assert tickers == sorted(["AAPL", "BRK-B", "BF-B", "MSFT"])
    assert all("." not in ticker for ticker in tickers)


def test_load_or_fetch_sp500_tickers_uses_cache_when_present(tmp_path):
    db_path = tmp_path / "test.duckdb"
    conn = storage.get_connection(db_path)
    storage.replace_sp500_tickers(conn, ["AAPL", "MSFT", "GOOGL"])
    conn.close()

    with patch("src.data.storage.config.DUCKDB_PATH", db_path):
        with patch("src.data.ingest.fetch_sp500_tickers") as fetch_mock:
            result = ingest.load_or_fetch_sp500_tickers()

    assert result == ["AAPL", "GOOGL", "MSFT"]  # sp500_tickers_cached returns them sorted
    fetch_mock.assert_not_called()


def test_load_or_fetch_sp500_tickers_fetches_and_caches_when_missing(tmp_path):
    db_path = tmp_path / "test.duckdb"
    fetched_tickers = ["AAPL", "MSFT", "GOOGL"]

    with patch("src.data.storage.config.DUCKDB_PATH", db_path):
        with patch("src.data.ingest.fetch_sp500_tickers", return_value=fetched_tickers) as fetch_mock:
            result = ingest.load_or_fetch_sp500_tickers()

    assert result == fetched_tickers
    fetch_mock.assert_called_once()

    conn = storage.get_connection(db_path)
    assert storage.sp500_tickers_cached(conn) == sorted(fetched_tickers)
    conn.close()


def test_load_or_fetch_prices_shares_cached_data_across_overlapping_ticker_sets(tmp_path):
    """The real payoff DuckDB storage exists for, replacing the old
    cache_name stopgap entirely: a request that overlaps an earlier one
    only fetches the genuinely new tickers, not the whole set again.
    """
    db_path = tmp_path / "test.duckdb"
    aapl_row = pd.DataFrame(
        {
            "date": ["2020-01-01"],
            "ticker": ["AAPL"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "adj_close": [100.0],
            "volume": [1000.0],
        }
    )
    msft_row = aapl_row.assign(ticker="MSFT", adj_close=200.0)

    with patch("src.data.storage.config.DUCKDB_PATH", db_path):
        with patch("src.data.ingest.fetch_price_history", return_value=aapl_row) as fetch_mock:
            first_result = ingest.load_or_fetch_prices(["AAPL"], "2020-01-01", "2020-01-01")
        fetch_mock.assert_called_once_with(["AAPL"], "2020-01-01", "2020-01-01")

        with patch("src.data.ingest.fetch_price_history", return_value=msft_row) as fetch_mock:
            second_result = ingest.load_or_fetch_prices(["AAPL", "MSFT"], "2020-01-01", "2020-01-01")
        fetch_mock.assert_called_once_with(["MSFT"], "2020-01-01", "2020-01-01")

    assert len(first_result) == 1
    assert set(second_result["ticker"]) == {"AAPL", "MSFT"}
