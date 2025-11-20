import pandas as pd
from fastapi.testclient import TestClient

from app.main import app, get_exchange, get_llm_client


class FakeExchange:
    def __init__(self):
        base_ts = pd.Timestamp("2024-01-01")
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
        self.ohlcv_1h = [
            [
                (base_ts + pd.Timedelta(hours=idx)).value // 10**6,
                100 + idx,
                100 + idx,
                100 + idx,
                100 + idx,
                200.0,
            ]
            for idx in range(130)
        ]
        self.order_book = {
            "bids": [[129.6, 10.0], [129.0, 5.0]],
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


class FakeLLM:
    def generate_signal(self, payload):
        return {
            "decision": "LONG",
            "confidence_score": 90,
            "entry_price": payload["timeframes"]["15m"]["ohlcv"][-1]["close"],
            "sl_price": payload["timeframes"]["15m"]["ohlcv"][-1]["close"] - 1.0,
            "tp_price": payload["timeframes"]["15m"]["ohlcv"][-1]["close"] + 3.0,
            "risk_reward_ratio": 3.0,
            "reasoning": "test llm output",
            "violations": [],
        }


def test_signal_endpoint_with_fakes():
    app.dependency_overrides[get_exchange] = lambda: FakeExchange()
    app.dependency_overrides[get_llm_client] = lambda: FakeLLM()
    client = TestClient(app)

    response = client.post("/signal", json={"symbol": "ETH/USDT", "sl_percentage": 1.0})
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "LONG"
    assert body["confidence_score"] == 90.0
    assert body["risk_reward_ratio"] == 3.0
    assert "15m" in body["timeframes"]
    assert len(body["timeframes"]["15m"]["ohlcv"]) == 50
    assert body["order_book"]["bid_volume"] > 0
    assert body["derivatives"]["funding_rate"] == 0.0001

    app.dependency_overrides = {}
