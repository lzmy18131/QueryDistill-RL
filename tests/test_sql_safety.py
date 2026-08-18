"""SQL AST safety layer (Layer 1) tests."""

from __future__ import annotations

import pytest

from querydistill.sql.safety import validate_sql

SAFE_CASES = [
    "SELECT a FROM t",
    "SELECT * FROM t WHERE a = 1",
    "WITH x AS (SELECT 1) SELECT * FROM x",
    "SELECT a FROM (SELECT a FROM t) sub",
    "SELECT 1 UNION SELECT 2",
    "SELECT 1; -- trailing comment",
    "SELECT 1 /* comment ; inside */",
    "SELECT * FROM t WHERE a = 1;",
]

UNSAFE_CASES = [
    "DROP TABLE users",
    "DELETE FROM users",
    "UPDATE users SET name = 'x'",
    "INSERT INTO users VALUES (1)",
    "REPLACE INTO users VALUES (1)",
    "ATTACH DATABASE 'x.db' AS x",
    "DETACH DATABASE x",
    "PRAGMA writable_schema = ON",
    "PRAGMA query_only = OFF",
    "VACUUM",
    "CREATE TABLE evil (x)",
    "CREATE TRIGGER trg AFTER INSERT ON users BEGIN SELECT 1; END",
    "SELECT load_extension('evil.dll')",
    "SELECT 1; DROP TABLE users",
    "SELECT 1 ; DROP TABLE users ; SELECT 2",
    "ALTER TABLE users ADD COLUMN x",
]


@pytest.mark.parametrize("sql", SAFE_CASES)
def test_safe_select_forms_are_allowed(sql):
    decision = validate_sql(sql)
    assert decision.safe, (sql, decision.reason)
    assert decision.error_type == "none"
    assert decision.statement_count == 1


@pytest.mark.parametrize("sql", UNSAFE_CASES)
def test_dangerous_statements_are_rejected(sql):
    decision = validate_sql(sql)
    assert not decision.safe, sql
    assert decision.error_type == "unsafe_sql", (sql, decision.reason)


def test_multiple_statement_semicolon_trick_is_unsafe():
    decision = validate_sql("SELECT 1; DROP TABLE users")
    assert not decision.safe
    assert decision.statement_count == 2
    assert "multiple statements" in decision.reason


def test_comment_wrapped_second_statement_is_single_statement():
    # The DROP is inside a comment: SQLite executes only the SELECT.
    decision = validate_sql("SELECT 1 /* ; DROP TABLE users */")
    assert decision.safe


def test_syntax_error_is_classified():
    decision = validate_sql("SELEC 1 FROM")
    assert not decision.safe
    assert decision.error_type == "syntax_error"


def test_empty_sql_is_format_error():
    decision = validate_sql("   ")
    assert not decision.safe
    assert decision.error_type == "format_error"


def test_decision_serializes():
    payload = validate_sql("DROP TABLE x").as_dict()
    assert payload["safe"] is False
    assert payload["error_type"] == "unsafe_sql"
    assert isinstance(payload["forbidden_nodes"], list)
