"""Error bucket classifier tests."""

from __future__ import annotations

import pytest

from querydistill.evaluation.errors import ErrorBucket, classify_error
from querydistill.outputs.parser import parse_model_output
from querydistill.sql.executor import ExecutionResult
from querydistill.sql.safety import validate_sql
from querydistill.sql.verifier import VerificationResult


def _exec(error_type="sqlite_error", message="", success=False, timed_out=False):
    return ExecutionResult(
        success=success, error_type=error_type, error_message=message, timed_out=timed_out
    )


def _equivalent():
    return VerificationResult(
        equivalent=True,
        strict_equivalent=True,
        partial_credit=False,
        kind="unordered_rows",
        reason="ok",
    )


def test_format_error():
    parsed = parse_model_output("no tags")
    assert classify_error(parsed, None, None, None) == ErrorBucket.FORMAT_ERROR


def test_syntax_error():
    parsed = parse_model_output("<sql>SELEC 1</sql>")
    safety = validate_sql(parsed.sql)
    assert classify_error(parsed, safety, None, None) == ErrorBucket.SYNTAX_ERROR


def test_unsafe_sql():
    parsed = parse_model_output("<sql>DROP TABLE x</sql>")
    safety = validate_sql(parsed.sql)
    assert classify_error(parsed, safety, None, None) == ErrorBucket.UNSAFE_SQL


def test_timeout():
    parsed = parse_model_output("<sql>SELECT 1</sql>")
    safety = validate_sql(parsed.sql)
    assert (
        classify_error(parsed, safety, _exec(error_type="timeout", timed_out=True), None)
        == ErrorBucket.TIMEOUT
    )


@pytest.mark.parametrize(
    ("message", "bucket"),
    [
        ("no such table: missing", ErrorBucket.WRONG_TABLE),
        ("no such column: nope", ErrorBucket.WRONG_COLUMN),
        ("misuse of aggregate: SUM()", ErrorBucket.AGGREGATION_ERROR),
        ("GROUP BY clause error", ErrorBucket.GROUP_BY_ERROR),
        ("ORDER BY term out of range", ErrorBucket.ORDER_ERROR),
        ("ambiguous column name: id", ErrorBucket.JOIN_ERROR),
        ("no join found", ErrorBucket.JOIN_ERROR),
        ("WHERE clause invalid", ErrorBucket.FILTER_ERROR),
        ("something schema-flavored", ErrorBucket.SCHEMA_ERROR),
    ],
)
def test_execution_error_message_buckets(message, bucket):
    parsed = parse_model_output("<sql>SELECT 1</sql>")
    safety = validate_sql(parsed.sql)
    assert classify_error(parsed, safety, _exec(message=message), None) == bucket


def test_wrong_result_vs_none():
    parsed = parse_model_output("<sql>SELECT 1</sql>")
    safety = validate_sql(parsed.sql)
    ok_execution = _exec(success=True)
    assert (
        classify_error(
            parsed,
            safety,
            ok_execution,
            VerificationResult(
                equivalent=False,
                strict_equivalent=False,
                partial_credit=False,
                kind="not_equivalent",
                reason="x",
            ),
        )
        == ErrorBucket.WRONG_RESULT
    )
    assert classify_error(parsed, safety, ok_execution, _equivalent()) == ErrorBucket.NONE


def test_all_buckets_are_declared():
    expected = {
        "syntax_error",
        "format_error",
        "unsafe_sql",
        "schema_error",
        "wrong_table",
        "wrong_column",
        "join_error",
        "filter_error",
        "aggregation_error",
        "group_by_error",
        "order_error",
        "timeout",
        "wrong_result",
        "none",
    }
    assert {bucket.value for bucket in ErrorBucket} == expected
