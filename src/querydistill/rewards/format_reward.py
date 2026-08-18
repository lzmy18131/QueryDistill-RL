"""Format reward: score the output protocol wrapper, not the SQL content."""

from __future__ import annotations

from ..outputs.parser import ParseResult

MALFORMED = -0.4
VALID = 0.05


def format_reward(parse_result: ParseResult, require_plan: bool = False) -> tuple[float, dict]:
    """Score protocol compliance.

    A beautifully formatted SQL that is wrong must not earn a high reward;
    this component alone never exceeds ``VALID``.
    """
    if parse_result.parse_error and parse_result.protocol != "fence":
        return MALFORMED, {"reason": parse_result.parse_error}
    if parse_result.protocol == "tags" and parse_result.format_ok:
        return VALID, {"reason": "single <sql> block parsed", "protocol": "tags"}
    if parse_result.protocol == "fence":
        return 0.0, {"reason": "markdown fence fallback (not the required protocol)"}
    return MALFORMED, {"reason": parse_result.parse_error or "format violation"}
