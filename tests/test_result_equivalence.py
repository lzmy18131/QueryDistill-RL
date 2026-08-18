"""ResultEquivalenceVerifier tests: multisets, ordering, NULL, tolerance, empty hacking."""

from __future__ import annotations

import pytest

from querydistill.sql.executor import ExecutionResult
from querydistill.sql.verifier import (
    ResultEquivalenceVerifier,
    extract_table_names,
    has_order_by,
)


def result(rows, columns=None, success=True, error_type="none", message="", truncated=False):
    return ExecutionResult(
        success=success,
        rows=rows,
        columns=columns or [f"c{i}" for i in range(len(rows[0]))] if rows else columns or ["c0"],
        error_type=error_type,
        error_message=message,
        row_count=len(rows),
        truncated=truncated,
    )


def test_unordered_rows_match_as_multiset():
    verifier = ResultEquivalenceVerifier()
    candidate = result([["a"], ["b"]], columns=["name"])
    gold = result([["b"], ["a"]], columns=["name"])
    verification = verifier.verify(candidate, gold, "SELECT name FROM t", "SELECT name FROM t")
    assert verification.equivalent
    assert verification.kind == "unordered_rows"


def test_duplicate_rows_matter():
    verifier = ResultEquivalenceVerifier()
    candidate = result([[1], [1], [2]], columns=["x"])
    gold = result([[1], [2], [2]], columns=["x"])
    verification = verifier.verify(candidate, gold)
    assert not verification.equivalent
    assert "multisets differ" in verification.reason


def test_nulls_are_distinct_from_null_strings():
    verifier = ResultEquivalenceVerifier()
    candidate = result([[None]], columns=["x"])
    gold = result([["NULL"]], columns=["x"])
    assert not verifier.verify(candidate, gold).equivalent
    assert verifier.verify(candidate, candidate).equivalent


def test_float_tolerance():
    verifier = ResultEquivalenceVerifier(float_tolerance=1e-6)
    candidate = result([[1.0], [2.0000001]], columns=["x"])
    gold = result([[1], [2.0]], columns=["x"])
    assert verifier.verify(candidate, gold).equivalent
    far = result([[1.01]], columns=["x"])
    assert not verifier.verify(far, result([[1.0]], columns=["x"])).equivalent


def test_int_float_equality_within_tolerance():
    verifier = ResultEquivalenceVerifier(float_tolerance=1e-6)
    assert verifier.verify(result([[1]], columns=["x"]), result([[1.0]], columns=["x"])).equivalent


def test_column_count_mismatch_fails():
    verifier = ResultEquivalenceVerifier()
    candidate = result([["a"]], columns=["c0"])
    gold = result([["a", "b"]], columns=["c0", "c1"])
    verification = verifier.verify(candidate, gold)
    assert not verification.equivalent
    assert "column" in verification.reason.lower()


def test_ordered_rows_preserve_order_when_gold_has_order_by():
    verifier = ResultEquivalenceVerifier()
    candidate = result([["a"], ["b"]], columns=["x"])
    gold = result([["b"], ["a"]], columns=["x"])
    ordered = verifier.verify(candidate, gold, "SELECT x FROM t", "SELECT x FROM t ORDER BY x")
    assert ordered.order_sensitive
    assert not ordered.equivalent
    assert "ordered mismatch" in ordered.reason

    same_order = verifier.verify(gold, gold, "SELECT x FROM t", "SELECT x FROM t ORDER BY x")
    assert same_order.equivalent
    assert same_order.kind == "ordered_rows"


def test_ordered_row_count_mismatch_fails():
    verifier = ResultEquivalenceVerifier()
    candidate = result([["a"], ["b"]], columns=["x"])
    gold = result([["a"]], columns=["x"])
    verification = verifier.verify(candidate, gold, None, "SELECT x FROM t ORDER BY x")
    assert not verification.equivalent


def test_failed_candidate_cannot_be_equivalent():
    verifier = ResultEquivalenceVerifier()
    candidate = result([], columns=["x"], success=False, error_type="sqlite_error", message="boom")
    gold = result([["a"]], columns=["x"])
    verification = verifier.verify(candidate, gold)
    assert not verification.equivalent
    assert "failed" in verification.reason


def test_both_empty_requires_structural_sanity():
    verifier = ResultEquivalenceVerifier()
    empty_candidate = result([], columns=["x"])
    empty_gold = result([], columns=["x"])
    tables = {"users"}

    # Constant SELECT 1 has no FROM -> rejected.
    constant = verifier.verify(
        empty_candidate,
        empty_gold,
        "SELECT 1 WHERE 1=0",
        "SELECT x FROM users WHERE 1=0",
        schema_tables=tables,
    )
    assert not constant.equivalent
    assert constant.kind == "empty_unverified"

    # Subset-of-gold tables + FROM present + known schema -> PARTIAL credit,
    # explicitly NOT strict correctness.
    structural = verifier.verify(
        empty_candidate,
        empty_gold,
        "SELECT x FROM users WHERE 1=0",
        "SELECT x FROM users WHERE 1=0",
        schema_tables=tables,
    )
    assert not structural.equivalent
    assert not structural.strict_equivalent
    assert structural.partial_credit
    assert structural.kind == "empty_structural_partial"


def test_both_empty_rejects_wrong_table():
    verifier = ResultEquivalenceVerifier()
    empty_candidate = result([], columns=["x"])
    empty_gold = result([], columns=["x"])
    verification = verifier.verify(
        empty_candidate,
        empty_gold,
        "SELECT x FROM other WHERE 1=0",
        "SELECT x FROM users WHERE 1=0",
        schema_tables={"users"},
    )
    assert not verification.equivalent


def test_extract_table_names_and_order_detection():
    assert "users" in extract_table_names(
        "SELECT * FROM users JOIN orders ON users.id = orders.uid"
    )
    assert extract_table_names("SELECT 1") == []
    assert has_order_by("SELECT * FROM t ORDER BY x") is True
    assert has_order_by("SELECT * FROM t") is False


def test_verification_serializes():
    verifier = ResultEquivalenceVerifier()
    payload = verifier.verify(
        result([["a"]], columns=["x"]), result([["a"]], columns=["x"])
    ).as_dict()
    assert payload["equivalent"] is True
    assert payload["kind"] == "unordered_rows"


def test_tolerance_constructor_validates():
    with pytest.raises(ValueError):
        ResultEquivalenceVerifier(float_tolerance=-0.1)


def test_truncated_candidate_never_strict_equivalent():
    verifier = ResultEquivalenceVerifier()
    candidate = result([["a"]], columns=["x"], truncated=True)
    gold = result([["a"]], columns=["x"])
    verification = verifier.verify(candidate, gold)
    assert not verification.strict_equivalent
    assert not verification.equivalent
    assert verification.kind == "truncated_unverified"


def test_truncated_gold_never_strict_equivalent():
    verifier = ResultEquivalenceVerifier()
    candidate = result([["a"]], columns=["x"])
    gold = result([["a"]], columns=["x"], truncated=True)
    verification = verifier.verify(candidate, gold)
    assert not verification.strict_equivalent
    assert verification.kind == "truncated_unverified"


def test_alias_difference_can_be_execution_equivalent():
    verifier = ResultEquivalenceVerifier()
    candidate = result([[3]], columns=["total"])
    gold = result([[3]], columns=["COUNT(*)"])
    verification = verifier.verify(
        candidate, gold, "SELECT COUNT(*) AS total FROM users", "SELECT COUNT(*) FROM users"
    )
    assert verification.strict_equivalent
    assert verification.equivalent
    assert verification.column_count_match
    assert verification.column_match is False


def test_column_count_mismatch_fails_even_with_alias_freedom():
    verifier = ResultEquivalenceVerifier()
    candidate = result([["a"]], columns=["x"])
    gold = result([["a", "b"]], columns=["x", "y"])
    verification = verifier.verify(candidate, gold)
    assert not verification.strict_equivalent
    assert not verification.column_count_match
