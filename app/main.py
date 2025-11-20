from typing import Optional

import ccxt
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.config import Settings, get_settings
from app.data import fetch_market_data
from app.llm import LLMClient

app = FastAPI(title="AI Futures Position Recommend Bot", version="0.1.0")


@app.get("/", tags=["system"])
def read_root() -> dict[str, str]:
    """Simple root endpoint for uptime checks."""
    return {"message": "AI Futures Position Recommend Bot is running"}


@app.get("/health", tags=["system"])
def health(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    """Health endpoint that surfaces minimal runtime info."""
    return {"status": "ok", "llm_model": settings.llm_model_name}


def get_exchange(settings: Settings = Depends(get_settings)):
    """Creates a Bybit exchange client."""
    exchange = ccxt.bybit(
        {
            "apiKey": settings.bybit_api_key,
            "secret": settings.bybit_api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        }
    )
    return exchange


def get_llm_client(settings: Settings = Depends(get_settings)) -> LLMClient:
    return LLMClient(settings)


class SignalRequest(BaseModel):
    symbol: str = Field(..., description="Trading pair, e.g., ETH/USDT:USDT")
    sl_percentage: float = Field(..., gt=0, description="Fixed stop loss percent, e.g., 1.0 for 1%")


class SignalResponse(BaseModel):
    decision: str
    confidence_score: float
    entry_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    reasoning: Optional[str] = None
    violations: list[str] = Field(default_factory=list)
    timeframes: dict
    order_book: dict
    derivatives: dict

    @field_validator("decision")
    @classmethod
    def validate_decision(cls, v: str):
        v_upper = v.upper()
        if v_upper not in {"LONG", "SHORT", "NEUTRAL"}:
            raise ValueError("decision must be LONG, SHORT, or NEUTRAL")
        return v_upper


@app.post("/signal", response_model=SignalResponse, tags=["signal"])
def generate_signal(
    body: SignalRequest,
    settings: Settings = Depends(get_settings),
    exchange=Depends(get_exchange),
    llm: LLMClient = Depends(get_llm_client),
):
    try:
        market_data = fetch_market_data(exchange, symbol=body.symbol, sl_percentage=body.sl_percentage)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch market data: {exc}")

    llm_input = {
        "symbol": body.symbol,
        "sl_percentage": body.sl_percentage,
        "timeframes": market_data["timeframes"],
        "order_book": market_data["order_book"],
        "derivatives": market_data["derivatives"],
    }

    try:
        sanitized = llm.generate_signal(llm_input)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {exc}")

    return SignalResponse(
        decision=sanitized["decision"],
        confidence_score=sanitized["confidence_score"],
        entry_price=sanitized["entry_price"],
        sl_price=sanitized["sl_price"],
        tp_price=sanitized["tp_price"],
        risk_reward_ratio=sanitized["risk_reward_ratio"],
        reasoning=sanitized.get("reasoning"),
        violations=sanitized.get("violations", []),
        timeframes=market_data["timeframes"],
        order_book=market_data["order_book"],
        derivatives=market_data["derivatives"],
    )
