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
    sanitized = sanitize_signal_response(raw, user_sl_pct=1.0)
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
    sanitized = sanitize_signal_response(raw, user_sl_pct=1.0)
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
    sanitized = sanitize_signal_response(raw, user_sl_pct=1.0)
    assert sanitized["decision"] == "NEUTRAL"
    assert "long_bounds" in sanitized["violations"]


def test_sl_adjustment_and_position_size_clamped():
    raw = {
        "decision": "LONG",
        "confidence_score": 90,
        "entry_price": 100.0,
        "sl_price": 99.0,
        "tp_price": 102.0,
        "risk_reward_ratio": 2.0,
        "adjusted_sl_percentage": 2.0,  # should clamp to 1.5 when user_sl_pct=1
    }
    sanitized = sanitize_signal_response(raw, user_sl_pct=1.0)
    assert sanitized["adjusted_sl_percentage"] == pytest.approx(1.5)
    assert sanitized["position_size_pct_of_equity"] == pytest.approx((1.0 / 1.5) * 100)
