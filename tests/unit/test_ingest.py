"""Guards the S&P 500 constituent sourcing: ticker normalisation for
yfinance compatibility, and the cache-hit/cache-miss pattern shared with
every other load_or_fetch_X function in ingest.py.
"""

from unittest.mock import patch

import pandas as pd

from src.data import ingest


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
    cached_tickers = ["AAPL", "MSFT", "GOOGL"]
    cache_path = tmp_path / "sp500_tickers.parquet"
    pd.DataFrame({"ticker": cached_tickers}).to_parquet(cache_path)

    with patch("src.data.ingest.config.CACHE_DIR", str(tmp_path)):
        with patch("src.data.ingest.fetch_sp500_tickers") as fetch_mock:
            result = ingest.load_or_fetch_sp500_tickers()

    assert result == cached_tickers
    fetch_mock.assert_not_called()


def test_load_or_fetch_sp500_tickers_fetches_and_caches_when_missing(tmp_path):
    fetched_tickers = ["AAPL", "MSFT", "GOOGL"]

    with patch("src.data.ingest.config.CACHE_DIR", str(tmp_path)):
        with patch("src.data.ingest.fetch_sp500_tickers", return_value=fetched_tickers) as fetch_mock:
            result = ingest.load_or_fetch_sp500_tickers()

    assert result == fetched_tickers
    fetch_mock.assert_called_once()
    assert (tmp_path / "sp500_tickers.parquet").exists()


def test_load_or_fetch_prices_with_distinct_cache_names_do_not_clobber_each_other(tmp_path):
    dev_prices = pd.DataFrame({"date": ["2020-01-01"], "ticker": ["AAPL"], "adj_close": [100.0]})
    sp500_prices = pd.DataFrame(
        {"date": ["2020-01-01"] * 2, "ticker": ["AAPL", "MSFT"], "adj_close": [100.0, 200.0]}
    )

    with patch("src.data.ingest.config.CACHE_DIR", str(tmp_path)):
        with patch("src.data.ingest.fetch_price_history", side_effect=[dev_prices, sp500_prices]):
            dev_result = ingest.load_or_fetch_prices(["AAPL"], "2020-01-01", "2020-01-01", cache_name="prices_dev")
            sp500_result = ingest.load_or_fetch_prices(
                ["AAPL", "MSFT"], "2020-01-01", "2020-01-01", cache_name="prices_sp500"
            )

        # Re-reading each by its own cache_name must return what was cached
        # under that name, not the other run's data.
        with patch("src.data.ingest.fetch_price_history") as fetch_mock:
            dev_reread = ingest.load_or_fetch_prices(["AAPL"], "2020-01-01", "2020-01-01", cache_name="prices_dev")
            sp500_reread = ingest.load_or_fetch_prices(
                ["AAPL", "MSFT"], "2020-01-01", "2020-01-01", cache_name="prices_sp500"
            )

    fetch_mock.assert_not_called()
    assert len(dev_result) == 1 and len(dev_reread) == 1
    assert len(sp500_result) == 2 and len(sp500_reread) == 2
    assert (tmp_path / "prices_dev.parquet").exists()
    assert (tmp_path / "prices_sp500.parquet").exists()


def test_tickers_with_complete_history_excludes_only_incomplete_tickers():
    dates = pd.bdate_range("2020-01-01", periods=5)
    wide_prices = pd.DataFrame(
        {
            "COMPLETE_A": [10.0, 11.0, 12.0, 13.0, 14.0],
            "COMPLETE_B": [20.0, 21.0, 22.0, 23.0, 24.0],
            "RECENT_IPO": [None, None, None, 30.0, 31.0],
        },
        index=dates,
    )

    result = ingest.tickers_with_complete_history(wide_prices)

    assert result == ["COMPLETE_A", "COMPLETE_B"]
    # The surviving tickers' own values are untouched by the incomplete
    # ticker's presence - this is the property that matters: one column's
    # gap must never corrupt another column's data.
    assert wide_prices["COMPLETE_A"].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]
