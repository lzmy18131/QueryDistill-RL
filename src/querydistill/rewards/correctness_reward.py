"""Correctness reward: the dominant, execution-equivalence based signal."""

from __future__ import annotations

from ..sql.verifier import VerificationResult

FULL_EQUIVALENT = 1.0
EMPTY_STRUCTURAL_EQUIVALENT = 0.25
NOT_EQUIVALENT = 0.0


def correctness_reward(verification: VerificationResult | None) -> tuple[float, dict]:
    """Map the result verifier onto the dominant reward component.

    Only STRICT equivalence earns the full correctness reward. Empty-empty
    structural results earn a small shaping credit (partial_credit=True) and
    never count as correct.
    """
    if verification is None:
        return 0.0, {"reason": "not verified"}
    if verification.strict_equivalent:
        return FULL_EQUIVALENT, {"reason": verification.reason, "kind": verification.kind}
    if verification.partial_credit:
        return EMPTY_STRUCTURAL_EQUIVALENT, {
            "reason": verification.reason,
            "kind": verification.kind,
            "confidence": "partial_shaping_only",
        }
    return NOT_EQUIVALENT, {
        "reason": verification.reason,
        "kind": verification.kind,
        "confidence": "none",
    }
