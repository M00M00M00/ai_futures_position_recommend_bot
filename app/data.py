from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import pandas_ta as ta


TimeframeKey = str


@dataclass
class TimeframeConfig:
    """Configuration for how many candles to fetch and how many to expose."""

    fetch_limit: int
    expose_limit: int


TIMEFRAME_CONFIG: dict[TimeframeKey, TimeframeConfig] = {
    # Fetch more than we expose so long lookbacks (e.g., SMA99) remain valid.
    "15m": TimeframeConfig(fetch_limit=130, expose_limit=50),
    "1h": TimeframeConfig(fetch_limit=180, expose_limit=20),
}


def ohlcv_to_dataframe(raw_ohlcv: Iterable[Iterable[Any]]) -> pd.DataFrame:
    df = pd.DataFrame(
        raw_ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    closes = df["close"]

    sma7 = closes.rolling(7).mean().iloc[-1]
    sma25 = closes.rolling(25).mean().iloc[-1]
    sma99 = closes.rolling(99).mean().iloc[-1]

    rsi_series = ta.rsi(closes, length=14)
    rsi = rsi_series.iloc[-1]

    bbands = ta.bbands(closes, length=20, std=2)
    macd = ta.macd(closes)  # defaults fast=12, slow=26, signal=9

    def _last_from_columns(frame: Optional[pd.DataFrame], prefix: str) -> Optional[float]:
        if frame is None:
            return None
        for col in frame.columns:
            if col.startswith(prefix):
                return _to_float_or_none(frame[col].iloc[-1])
        return None

    bollinger_slice = {
        "upper": _last_from_columns(bbands, "BBU_"),
        "lower": _last_from_columns(bbands, "BBL_"),
        "percent_b": _last_from_columns(bbands, "BBP_"),
    }

    macd_slice = {
        "macd": _last_from_columns(macd, "MACD_"),
        "signal": _last_from_columns(macd, "MACDs_"),
        "histogram": _last_from_columns(macd, "MACDh_"),
    }

    return {
        "sma": {"7": _to_float_or_none(sma7), "25": _to_float_or_none(sma25), "99": _to_float_or_none(sma99)},
        "rsi": _to_float_or_none(rsi),
        "bollinger_bands": bollinger_slice,
        "macd": macd_slice,
    }


def dataframe_to_records(df: pd.DataFrame, limit: int) -> List[Dict[str, Any]]:
    trimmed = df.tail(limit)
    return [
        {
            "timestamp": row["timestamp"].isoformat(),
            "open": _to_float_or_none(row["open"]),
            "high": _to_float_or_none(row["high"]),
            "low": _to_float_or_none(row["low"]),
            "close": _to_float_or_none(row["close"]),
            "volume": _to_float_or_none(row["volume"]),
        }
        for _, row in trimmed.iterrows()
    ]


def aggregate_order_book(order_book: Dict[str, List[List[float]]], mid_price: float, window_pct: float = 0.5) -> Dict[str, float]:
    lower = mid_price * (1 - window_pct / 100)
    upper = mid_price * (1 + window_pct / 100)

    bid_volume = sum(amount for price, amount in order_book.get("bids", []) if price >= lower)
    ask_volume = sum(amount for price, amount in order_book.get("asks", []) if price <= upper)

    imbalance_ratio = None
    if ask_volume:
        imbalance_ratio = bid_volume / ask_volume
    elif bid_volume:
        imbalance_ratio = float("inf")

    return {"bid_volume": bid_volume, "ask_volume": ask_volume, "window_pct": window_pct, "imbalance_ratio": imbalance_ratio}


def _first_column_with_prefix(frame: Optional[pd.DataFrame], prefix: str) -> Optional[str]:
    if frame is None:
        return None
    for col in frame.columns:
        if col.startswith(prefix):
            return col
    return None


def _extract_open_interest_value(point: Dict[str, Any]) -> Optional[float]:
    for key in ("openInterestAmount", "openInterestValue", "oi", "open_interest"):
        if key in point:
            return _to_float_or_none(point[key])
    return None


def fetch_derivatives(exchange: Any, symbol: str) -> Dict[str, Optional[float]]:
    try:
        funding = exchange.fetch_funding_rate(symbol)
        funding_rate = _to_float_or_none(funding.get("fundingRate"))
    except Exception:
        funding_rate = None

    try:
        oi_history = exchange.fetch_open_interest_history(symbol, timeframe="1h", limit=2)
    except Exception:
        oi_history = []

    oi_change_pct: Optional[float] = None
    if oi_history and len(oi_history) >= 2:
        prev = _extract_open_interest_value(oi_history[-2])
        last = _extract_open_interest_value(oi_history[-1])
        if prev not in (None, 0):
            oi_change_pct = ((last - prev) / prev) * 100 if last is not None else None

    return {"funding_rate": funding_rate, "open_interest_change_pct": oi_change_pct}


def fetch_market_data(exchange: Any, symbol: str, sl_percentage: float) -> Dict[str, Any]:
    timeframe_payload: Dict[str, Any] = {}

    for tf, cfg in TIMEFRAME_CONFIG.items():
        raw = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=cfg.fetch_limit)
        df = ohlcv_to_dataframe(raw)
        indicators = compute_indicators(df)
        ohlcv_records = dataframe_to_records(df, cfg.expose_limit)
        timeframe_payload[tf] = {"ohlcv": ohlcv_records, "indicators": indicators}

    # use the most recent 15m close as mid-price for depth aggregation
    reference_close = timeframe_payload["15m"]["ohlcv"][-1]["close"]
    order_book = exchange.fetch_order_book(symbol)
    depth = {
        "windows": {
            "0.5": aggregate_order_book(order_book, reference_close, window_pct=0.5),
            "1.0": aggregate_order_book(order_book, reference_close, window_pct=1.0),
        }
    }
    derivatives = fetch_derivatives(exchange, symbol)

    # market context to reduce LLM load on simple classifications
    market_context: Dict[str, Any] = {}
    close_1h = timeframe_payload["1h"]["ohlcv"][-1]["close"]
    sma99_1h = timeframe_payload["1h"]["indicators"]["sma"]["99"]
    if sma99_1h is not None:
        if close_1h > sma99_1h:
            market_context["current_price_vs_sma99_1h"] = "ABOVE"
        elif close_1h < sma99_1h:
            market_context["current_price_vs_sma99_1h"] = "BELOW"
        else:
            market_context["current_price_vs_sma99_1h"] = "NEAR"

    bb_1h = timeframe_payload["1h"]["indicators"]["bollinger_bands"]
    if bb_1h["upper"] is not None and bb_1h["lower"] is not None:
        # compute simple volatility state using BB width vs rolling median width on 1h
        raw = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=min(TIMEFRAME_CONFIG["1h"].fetch_limit, 180))
        df = ohlcv_to_dataframe(raw)
        closes = df["close"]
        bbands_full = ta.bbands(closes, length=20, std=2)
        if bbands_full is not None:
            upper_col = _first_column_with_prefix(bbands_full, "BBU_")
            lower_col = _first_column_with_prefix(bbands_full, "BBL_")
            if upper_col and lower_col:
                widths = bbands_full[upper_col] - bbands_full[lower_col]
                if len(widths.dropna()) >= 20:
                    last_width = widths.iloc[-1]
                    median_width = widths.dropna().rolling(20).median().iloc[-1]
                    if median_width and last_width:
                        if last_width > median_width * 1.1:
                            market_context["volatility_state"] = "EXPANSION"
                        elif last_width < median_width * 0.9:
                            market_context["volatility_state"] = "SQUEEZE"
                        else:
                            market_context["volatility_state"] = "NEUTRAL"

    return {
        "symbol": symbol,
        "sl_percentage": sl_percentage,
        "timeframes": timeframe_payload,
        "order_book": depth,
        "derivatives": derivatives,
        "market_context": market_context,
    }
