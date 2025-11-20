from __future__ import annotations

from typing import Any, Dict, Literal, Optional


Decision = Literal["LONG", "SHORT", "NEUTRAL"]


class SignalValidationError(Exception):
    pass


def compute_risk_reward(decision: Decision, entry: float, sl: float, tp: float) -> Optional[float]:
    if decision == "LONG":
        risk = entry - sl
        reward = tp - entry
    elif decision == "SHORT":
        risk = sl - entry
        reward = entry - tp
    else:
        return None

    if risk <= 0 or reward <= 0:
        return None
    return reward / risk


def sanitize_signal_response(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Validates and applies guardrails to the LLM output according to spec."""
    confidence_threshold = 70.0
    decision = str(raw.get("decision", "")).upper()
    confidence = raw.get("confidence_score")
    entry = raw.get("entry_price")
    sl = raw.get("sl_price")
    tp = raw.get("tp_price")
    rr = raw.get("risk_reward_ratio")
    reasoning = raw.get("reasoning")

    violations = []

    if decision not in {"LONG", "SHORT", "NEUTRAL"}:
        violations.append("invalid_decision")
        decision = "NEUTRAL"

    # confidence check
    try:
        confidence_val = float(confidence)
    except (TypeError, ValueError):
        confidence_val = 0.0
    if confidence_val < confidence_threshold:
        decision = "NEUTRAL"

    if decision != "NEUTRAL":
        # numeric ordering constraints
        try:
            entry_f = float(entry)
            sl_f = float(sl)
            tp_f = float(tp)
        except (TypeError, ValueError):
            violations.append("invalid_numbers")
            decision = "NEUTRAL"
        else:
            if decision == "LONG":
                if not (sl_f < entry_f < tp_f):
                    violations.append("long_bounds")
                    decision = "NEUTRAL"
            elif decision == "SHORT":
                if not (tp_f < entry_f < sl_f):
                    violations.append("short_bounds")
                    decision = "NEUTRAL"

            if decision != "NEUTRAL":
                rr_calc = compute_risk_reward(decision, entry_f, sl_f, tp_f)
                rr_final = rr_calc
                if rr_final is None:
                    violations.append("rr_calc")
                    decision = "NEUTRAL"
                else:
                    if rr_final < 1 or rr_final > 3:
                        violations.append("rr_range")
                        decision = "NEUTRAL"
                rr = rr_final
            else:
                rr = None
    else:
        rr = None

    sanitized = {
        "decision": decision,
        "confidence_score": confidence_val,
        "entry_price": entry if decision != "NEUTRAL" else None,
        "sl_price": sl if decision != "NEUTRAL" else None,
        "tp_price": tp if decision != "NEUTRAL" else None,
        "risk_reward_ratio": rr,
        "reasoning": reasoning,
        "violations": violations,
    }
    return sanitized
