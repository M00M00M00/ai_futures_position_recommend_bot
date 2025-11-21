from bot.main import build_embed
from bot.main import normalize_symbol


def test_build_embed_formats_fields():
    signal = {
        "decision": "LONG",
        "confidence_score": 85.5,
        "entry_price": 100.0,
        "sl_price": 99.0,
        "tp_price": 103.0,
        "risk_reward_ratio": 3.0,
        "reasoning": "test reasoning",
    }
    embed = build_embed(signal, symbol="ETH/USDT", sl_percentage=1.0)
    assert embed.title == "Signal Analysis: ETH/USDT"
    field_names = [f.name for f in embed.fields]
    assert "Decision" in field_names
    assert "Confidence" in field_names
    assert "R:R" in field_names
    assert "Stop Loss" in field_names
    assert "Take Profit" in field_names


def test_normalize_symbol_adds_usdt_suffix():
    assert normalize_symbol("eth") == "ETH/USDT:USDT"
    assert normalize_symbol("BTC") == "BTC/USDT:USDT"
    assert normalize_symbol("ETH/USDT:USDT") == "ETH/USDT:USDT"


def test_reasoning_truncated_to_discord_limit():
    long_reasoning = "x" * 1500
    signal = {
        "decision": "LONG",
        "confidence_score": 85.5,
        "entry_price": 100.0,
        "sl_price": 99.0,
        "tp_price": 103.0,
        "risk_reward_ratio": 3.0,
        "reasoning": long_reasoning,
    }
    embed = build_embed(signal, symbol="ETH/USDT", sl_percentage=1.0)
    assert embed.description
    assert len(embed.description) <= 4000
