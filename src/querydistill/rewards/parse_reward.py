"""Parse reward: the extracted SQL must be a valid single statement."""

from __future__ import annotations

from ..outputs.parser import ParseResult
from ..sql.safety import validate_sql

VALID = 0.05
INVALID = -0.4


def parse_reward(parse_result: ParseResult) -> tuple[float, dict]:
    if parse_result.sql is None:
        return INVALID, {"reason": "no SQL extracted"}
    decision = validate_sql(parse_result.sql)
    if decision.safe:
        return VALID, {"reason": decision.reason}
    if decision.error_type == "syntax_error":
        return INVALID, {"reason": decision.reason, "error_type": "syntax_error"}
    # Unsafe-but-parseable statements are scored here as unparseable for the
    # policy and are additionally hard-penalized by the safety component.
    return INVALID, {"reason": decision.reason, "error_type": decision.error_type}
