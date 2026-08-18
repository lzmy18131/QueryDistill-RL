"""Safety reward: hard penalty for anything the safety layers reject."""

from __future__ import annotations

from ..sql.safety import SafetyDecision

SAFE = 0.05
UNSAFE = -1.0


def safety_reward(decision: SafetyDecision | None) -> tuple[float, dict]:
    if decision is None:
        return 0.0, {"reason": "safety check not performed"}
    if decision.safe:
        return SAFE, {"reason": decision.reason}
    return UNSAFE, {
        "reason": decision.reason,
        "error_type": decision.error_type,
        "forbidden_nodes": decision.forbidden_nodes,
    }
