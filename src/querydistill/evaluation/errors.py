"""Error bucket classification for per-example failure analysis."""

from __future__ import annotations

from enum import StrEnum

from ..outputs.parser import ParseResult
from ..sql.executor import ExecutionResult
from ..sql.safety import SafetyDecision
from ..sql.verifier import VerificationResult


class ErrorBucket(StrEnum):
    NONE = "none"
    SYNTAX_ERROR = "syntax_error"
    FORMAT_ERROR = "format_error"
    UNSAFE_SQL = "unsafe_sql"
    SCHEMA_ERROR = "schema_error"
    WRONG_TABLE = "wrong_table"
    WRONG_COLUMN = "wrong_column"
    JOIN_ERROR = "join_error"
    FILTER_ERROR = "filter_error"
    AGGREGATION_ERROR = "aggregation_error"
    GROUP_BY_ERROR = "group_by_error"
    ORDER_ERROR = "order_error"
    TIMEOUT = "timeout"
    WRONG_RESULT = "wrong_result"


def _keyword_bucket(message: str) -> ErrorBucket | None:
    lowered = message.lower()
    rules = [
        ("no such table", ErrorBucket.WRONG_TABLE),
        ("no such column", ErrorBucket.WRONG_COLUMN),
        ("misuse of aggregate", ErrorBucket.AGGREGATION_ERROR),
        ("group by", ErrorBucket.GROUP_BY_ERROR),
        ("order by", ErrorBucket.ORDER_ERROR),
        ("ambiguous", ErrorBucket.JOIN_ERROR),
        ("join", ErrorBucket.JOIN_ERROR),
        ("where", ErrorBucket.FILTER_ERROR),
    ]
    for keyword, bucket in rules:
        if keyword in lowered:
            return bucket
    return None


def classify_error(
    parse_result: ParseResult | None,
    safety: SafetyDecision | None,
    execution: ExecutionResult | None,
    verification: VerificationResult | None = None,
) -> ErrorBucket:
    if parse_result is not None and parse_result.sql is None:
        return ErrorBucket.FORMAT_ERROR
    if safety is not None and safety.error_type == "syntax_error":
        return ErrorBucket.SYNTAX_ERROR
    if safety is not None and not safety.safe:
        return ErrorBucket.UNSAFE_SQL
    if execution is None:
        return ErrorBucket.FORMAT_ERROR
    if execution.timed_out or execution.error_type == "timeout":
        return ErrorBucket.TIMEOUT
    if not execution.success:
        return _keyword_bucket(execution.error_message) or ErrorBucket.SCHEMA_ERROR
    if verification is None or not verification.equivalent:
        return ErrorBucket.WRONG_RESULT
    return ErrorBucket.NONE
