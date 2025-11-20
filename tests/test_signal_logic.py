import pytest

from app.signal_logic import compute_risk_reward, sanitize_signal_response


def test_sanitize_valid_long_passes_through():
    raw = {
        "decision": "LONG",
        "confidence_score": 90,
        "entry_price": 100.0,
        "sl_price": 99.0,
        "tp_price": 102.0,
        "risk_reward_ratio": 2.0,  # will be recalculated
        "reasoning": "test",
    }
    sanitized = sanitize_signal_response(raw)
    assert sanitized["decision"] == "LONG"
    assert sanitized["risk_reward_ratio"] == pytest.approx(2.0)
    assert sanitized["violations"] == []


def test_confidence_below_threshold_forces_neutral():
    raw = {
        "decision": "LONG",
        "confidence_score": 65,  # below threshold 70
        "entry_price": 100,
        "sl_price": 99,
        "tp_price": 103,
        "risk_reward_ratio": 4.0,
    }
    sanitized = sanitize_signal_response(raw)
    assert sanitized["decision"] == "NEUTRAL"
    assert sanitized["risk_reward_ratio"] is None


def test_invalid_bounds_turns_neutral():
    raw = {
        "decision": "LONG",
        "confidence_score": 90,
        "entry_price": 100,
        "sl_price": 101,  # invalid for long
        "tp_price": 102,
    }
    sanitized = sanitize_signal_response(raw)
    assert sanitized["decision"] == "NEUTRAL"
    assert "long_bounds" in sanitized["violations"]
