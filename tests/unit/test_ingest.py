"""Guards the S&P 500 constituent sourcing: ticker normalisation for
yfinance compatibility, that load_or_fetch_X correctly delegates caching to
src.data.storage (DuckDB) rather than duplicating that logic, and that
price/dividend fetches retry with backoff on a rate limit.
"""

from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest
from yfinance.exceptions import YFRateLimitError

from src import config
from src.data import ingest, storage


def _fake_wikipedia_response(tickers: list[str], sectors: list[str] = None) -> str:
    """A minimal HTML table shaped like Wikipedia's real constituent table -
    just enough for pd.read_html(..., match="Symbol") to find it.
    """
    sectors = sectors or ["Information Technology"] * len(tickers)
    table = pd.DataFrame({"Symbol": tickers, "Security": [f"Company {t}" for t in tickers], "GICS Sector": sectors})
    return table.to_html(index=False)


def _fake_response(html: str):
    class _FakeResponse:
        text = html

        def raise_for_status(self):
            pass

    return _FakeResponse()


def test_fetch_sp500_tickers_normalises_share_classes_for_yfinance():
    html = _fake_wikipedia_response(["AAPL", "BRK.B", "BF.B", "MSFT"])

    with patch("src.data.ingest.requests.get", return_value=_fake_response(html)):
        tickers = ingest.fetch_sp500_tickers()

    assert tickers == sorted(["AAPL", "BRK-B", "BF-B", "MSFT"])
    assert all("." not in ticker for ticker in tickers)


def test_fetch_sp500_constituents_captures_ticker_and_sector():
    html = _fake_wikipedia_response(["AAPL", "JPM"], sectors=["Information Technology", "Financials"])

    with patch("src.data.ingest.requests.get", return_value=_fake_response(html)):
        constituents = ingest.fetch_sp500_constituents()

    assert dict(zip(constituents["ticker"], constituents["sector"])) == {
        "AAPL": "Information Technology",
        "JPM": "Financials",
    }


def test_load_or_fetch_sp500_tickers_uses_cache_when_present(tmp_path):
    db_path = tmp_path / "test.duckdb"
    conn = storage.get_connection(db_path)
    storage.replace_sp500_constituents(
        conn, pd.DataFrame({"ticker": ["AAPL", "MSFT", "GOOGL"], "sector": ["Information Technology"] * 3})
    )
    conn.close()

    with patch("src.data.storage.config.DUCKDB_PATH", db_path):
        with patch("src.data.ingest.fetch_sp500_constituents") as fetch_mock:
            result = ingest.load_or_fetch_sp500_tickers()

    assert result == ["AAPL", "GOOGL", "MSFT"]  # sp500_tickers_cached returns them sorted
    fetch_mock.assert_not_called()


def test_load_or_fetch_sp500_tickers_fetches_and_caches_when_missing(tmp_path):
    db_path = tmp_path / "test.duckdb"
    fetched_constituents = pd.DataFrame(
        {"ticker": ["AAPL", "MSFT", "GOOGL"], "sector": ["Information Technology"] * 3}
    )

    with patch("src.data.storage.config.DUCKDB_PATH", db_path):
        with patch("src.data.ingest.fetch_sp500_constituents", return_value=fetched_constituents) as fetch_mock:
            result = ingest.load_or_fetch_sp500_tickers()

    assert result == ["AAPL", "MSFT", "GOOGL"]
    fetch_mock.assert_called_once()

    conn = storage.get_connection(db_path)
    assert storage.sp500_tickers_cached(conn) == sorted(result)
    conn.close()


def test_load_or_fetch_sp500_sectors_uses_cache_when_present(tmp_path):
    db_path = tmp_path / "test.duckdb"
    conn = storage.get_connection(db_path)
    storage.replace_sp500_constituents(
        conn, pd.DataFrame({"ticker": ["AAPL", "JPM"], "sector": ["Information Technology", "Financials"]})
    )
    conn.close()

    with patch("src.data.storage.config.DUCKDB_PATH", db_path):
        with patch("src.data.ingest.fetch_sp500_constituents") as fetch_mock:
            sectors = ingest.load_or_fetch_sp500_sectors(["AAPL", "JPM"])

    assert sectors == {"AAPL": "Information Technology", "JPM": "Financials"}
    fetch_mock.assert_not_called()


def test_load_or_fetch_sp500_sectors_fetches_and_caches_when_missing(tmp_path):
    db_path = tmp_path / "test.duckdb"
    fetched_constituents = pd.DataFrame({"ticker": ["AAPL", "JPM"], "sector": ["Information Technology", "Financials"]})

    with patch("src.data.storage.config.DUCKDB_PATH", db_path):
        with patch("src.data.ingest.fetch_sp500_constituents", return_value=fetched_constituents) as fetch_mock:
            sectors = ingest.load_or_fetch_sp500_sectors(["AAPL", "JPM"])

    assert sectors == {"AAPL": "Information Technology", "JPM": "Financials"}
    fetch_mock.assert_called_once()


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


def test_download_with_retry_retries_and_recovers_from_a_rate_limit():
    fake_result = pd.DataFrame({"Close": [1.0]})

    with patch(
        "src.data.ingest.yf.download", side_effect=[YFRateLimitError(), YFRateLimitError(), fake_result]
    ) as download_mock, patch("src.data.ingest.time.sleep") as sleep_mock:
        result = ingest._download_with_retry(["AAPL"], date(2020, 1, 1), date(2020, 12, 31))

    assert result is fake_result
    assert download_mock.call_count == 3
    # Exponential backoff: base * 2**0, then base * 2**1.
    assert sleep_mock.call_args_list == [
        ((config.YFINANCE_DOWNLOAD_BACKOFF_BASE_SECONDS,),),
        ((config.YFINANCE_DOWNLOAD_BACKOFF_BASE_SECONDS * 2,),),
    ]


def test_download_with_retry_raises_after_exhausting_retries():
    with patch("src.data.ingest.yf.download", side_effect=YFRateLimitError()), patch(
        "src.data.ingest.time.sleep"
    ):
        with pytest.raises(YFRateLimitError):
            ingest._download_with_retry(["AAPL"], date(2020, 1, 1), date(2020, 12, 31))


def test_fetch_price_history_uses_the_retrying_batched_download():
    dates = pd.to_datetime(["2023-01-03"]).rename("Date")
    columns = pd.MultiIndex.from_tuples([("Close", "AAPL")], names=["Price", "Ticker"])
    wide = pd.DataFrame([[100.0]], index=dates, columns=columns)

    with patch("src.data.ingest._download_with_retry", return_value=wide) as download_mock:
        ingest.fetch_price_history(["AAPL"], date(2023, 1, 1), date(2023, 1, 31))

    download_mock.assert_called_once_with(
        ["AAPL"], date(2023, 1, 1), date(2023, 1, 31), auto_adjust=False
    )


def test_fetch_dividends_filters_zero_padding_and_reshapes_to_long_format():
    """yfinance pads every non-payment day with 0.0 in the Dividends column -
    sourced from the same batched download as prices (see
    _download_with_retry's docstring for why), not the far more
    rate-limit-prone yf.Ticker(...).dividends per-ticker endpoint.
    """
    dates = pd.to_datetime(["2023-01-03", "2023-01-04"])
    columns = pd.MultiIndex.from_tuples(
        [("Dividends", "AAPL"), ("Dividends", "MSFT")], names=["Price", "Ticker"]
    )
    wide = pd.DataFrame([[0.24, 0.0], [0.0, 0.0]], index=dates, columns=columns)

    with patch("src.data.ingest._download_with_retry", return_value=wide) as download_mock:
        result = ingest.fetch_dividends(["AAPL", "MSFT"], date(2023, 1, 1), date(2023, 1, 31))

    download_mock.assert_called_once_with(
        ["AAPL", "MSFT"], date(2023, 1, 1), date(2023, 1, 31), actions=True, auto_adjust=False
    )
    assert result.to_dict("records") == [
        {"date": pd.Timestamp("2023-01-03"), "ticker": "AAPL", "dividend_amount": 0.24}
    ]
