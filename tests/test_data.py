import pandas as pd
import pytest

from app import data as market_data


def test_compute_indicators_on_monotonic_series_produces_expected_sma():
    rows = []
    base_ts = pd.Timestamp("2024-01-01")
    for idx in range(120):
        price = idx + 1
        rows.append(
            [
                (base_ts + pd.Timedelta(minutes=15 * idx)).value // 10**6,
                price,
                price,
                price,
                price,
                100.0,
            ]
        )

    df = market_data.ohlcv_to_dataframe(rows)
    indicators = market_data.compute_indicators(df)

    assert indicators["sma"]["7"] == 117.0  # avg of 114..120
    assert indicators["sma"]["25"] == 108.0  # avg of 96..120
    assert indicators["sma"]["99"] == 71.0  # avg of 22..120
    assert 0 <= indicators["rsi"] <= 100
    assert set(indicators["bollinger_bands"].keys()) == {"upper", "lower", "percent_b"}
    assert set(indicators["macd"].keys()) == {"macd", "signal", "histogram"}


class FakeExchange:
    id = "bybit"

    def __init__(self):
        base_ts = pd.Timestamp("2024-01-01")
        # 15m series 130 points; 1..130 prices
        self.ohlcv_15m = [
            [
                (base_ts + pd.Timedelta(minutes=15 * idx)).value // 10**6,
                idx + 1,
                idx + 1,
                idx + 1,
                idx + 1,
                100.0,
            ]
            for idx in range(130)
        ]
        # 1h series 180 points; 100..279 prices
        self.ohlcv_1h = [
            [
                (base_ts + pd.Timedelta(hours=idx)).value // 10**6,
                100 + idx,
                100 + idx,
                100 + idx,
                100 + idx,
                200.0,
            ]
            for idx in range(180)
        ]

        self.order_book = {
            "bids": [[129.6, 10.0], [129.0, 5.0], [128.0, 7.0]],
            "asks": [[130.4, 12.0], [130.8, 6.0]],
        }
        self.funding_rate = 0.0001
        self.oi_history = [
            {"openInterestAmount": 1000.0},
            {"openInterestAmount": 1100.0},
        ]

    def fetch_ohlcv(self, symbol, timeframe, limit):
        if timeframe == "15m":
            return self.ohlcv_15m[:limit]
        if timeframe == "1h":
            return self.ohlcv_1h[:limit]
        raise ValueError("unsupported timeframe")

    def fetch_order_book(self, symbol):
        return self.order_book

    def fetch_funding_rate(self, symbol):
        return {"fundingRate": self.funding_rate}

    def fetch_open_interest_history(self, symbol, timeframe="1h", since=None, limit=None):
        return self.oi_history


def test_fetch_market_data_shapes_clusters_and_calculates_depth_and_derivatives():
    exchange = FakeExchange()
    result = market_data.fetch_market_data(exchange, symbol="ETH/USDT", sl_percentage=1.0)

    assert result["symbol"] == "ETH/USDT"
    assert result["sl_percentage"] == 1.0

    tf15 = result["timeframes"]["15m"]
    assert len(tf15["ohlcv"]) == 50  # expose limit
    assert tf15["ohlcv"][-1]["close"] == 130.0
    assert tf15["indicators"]["sma"]["7"] == 127.0  # avg of 124..130
    assert tf15["indicators"]["sma"]["25"] == 118.0  # avg of 106..130
    assert tf15["indicators"]["sma"]["99"] == 81.0  # avg of 32..130

    tf1h = result["timeframes"]["1h"]
    assert len(tf1h["ohlcv"]) == 20
    assert tf1h["ohlcv"][-1]["close"] == 279.0
    assert tf1h["indicators"]["sma"]["99"] == pytest.approx(230.0)

    depth = result["order_book"]["windows"]
    assert depth["0.5"]["bid_volume"] == 10.0  # only 129.6 within 0.5% window of 130
    assert depth["0.5"]["ask_volume"] == 12.0  # only 130.4 within window
    assert depth["1.0"]["bid_volume"] >= depth["0.5"]["bid_volume"]
    assert depth["1.0"]["ask_volume"] >= depth["0.5"]["ask_volume"]

    derivatives = result["derivatives"]
    assert derivatives["funding_rate"] == exchange.funding_rate
    assert derivatives["open_interest_change_pct"] == pytest.approx(10.0)
