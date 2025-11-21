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


def sanitize_signal_response(
    raw: Dict[str, Any],
    user_sl_pct: float,
    *,
    min_sl_factor: float = 0.5,
    max_sl_factor: float = 1.5,
    min_position_size_pct: float = 50.0,
    max_position_size_pct: float = 150.0,
) -> Dict[str, Any]:
    """Validates and applies guardrails to the LLM output according to spec."""
    confidence_threshold = raw.get("confidence_threshold", 70.0)
    decision = str(raw.get("decision", "")).upper()
    confidence = raw.get("confidence_score")
    entry = raw.get("entry_price")
    sl = raw.get("sl_price")
    tp = raw.get("tp_price")
    rr = raw.get("risk_reward_ratio")
    reasoning = raw.get("reasoning")
    sl_adj_pct_raw = raw.get("adjusted_sl_percentage")

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
                    if rr_final < 1 or rr_final > 10:
                        violations.append("rr_range")
                        decision = "NEUTRAL"
                rr = rr_final
            else:
                rr = None
    else:
        rr = None

    # SL adjustment and position sizing to maintain similar equity risk
    base_sl_pct = max(user_sl_pct, 0.01)
    try:
        sl_adj_pct_candidate = float(sl_adj_pct_raw) if sl_adj_pct_raw is not None else base_sl_pct
    except (TypeError, ValueError):
        sl_adj_pct_candidate = base_sl_pct

    lower_bound = base_sl_pct * min_sl_factor
    upper_bound = base_sl_pct * max_sl_factor
    adjusted_sl_pct = min(max(sl_adj_pct_candidate, lower_bound), upper_bound)

    position_size_pct = None
    if decision != "NEUTRAL":
        position_size_pct = (base_sl_pct / adjusted_sl_pct) * 100.0
        position_size_pct = min(max(position_size_pct, min_position_size_pct), max_position_size_pct)
    else:
        adjusted_sl_pct = None
        position_size_pct = None

    sanitized = {
        "decision": decision,
        "confidence_score": confidence_val,
        "entry_price": entry if decision != "NEUTRAL" else None,
        "sl_price": sl if decision != "NEUTRAL" else None,
        "tp_price": tp if decision != "NEUTRAL" else None,
        "risk_reward_ratio": rr,
        "reasoning": reasoning,
        "adjusted_sl_percentage": adjusted_sl_pct,
        "position_size_pct_of_equity": position_size_pct,
        "violations": violations,
    }
    return sanitized
