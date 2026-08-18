"""SafeSQLExecutor tests: authorizer, process isolation, timeout, max_rows."""

from __future__ import annotations

import sqlite3
import time

import pytest

from querydistill.sql.executor import (
    SafeSQLExecutor,
    _authorizer,
    _read_only_uri,
)


@pytest.fixture()
def executor(tiny_db):
    return SafeSQLExecutor(tiny_db, max_rows=100, max_execution_ms=1000)


def test_select_executes_and_returns_rows(executor):
    result = executor.execute("SELECT name FROM users ORDER BY id")
    assert result.success
    assert result.columns == ["name"]
    assert result.rows == [["Alice"], ["Bob"], ["Carol"]]
    assert result.row_count == 3
    assert result.timed_out is False


def test_with_select_executes(executor):
    result = executor.execute(
        "WITH young AS (SELECT * FROM users WHERE age < 30) SELECT name FROM young"
    )
    assert result.success
    assert result.rows == [["Bob"]]


def test_nested_and_union_selects_execute(executor):
    nested = executor.execute("SELECT * FROM (SELECT name FROM users) ORDER BY name")
    assert nested.success and nested.row_count == 3
    union = executor.execute("SELECT name FROM users UNION SELECT name FROM users")
    assert union.success and union.row_count == 3


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE users",
        "DELETE FROM users",
        "UPDATE users SET name='x'",
        "INSERT INTO users VALUES (4, 'D', 40)",
        "ATTACH DATABASE 'x.db' AS x",
        "DETACH DATABASE x",
        "PRAGMA writable_schema = ON",
        "VACUUM",
        "CREATE TABLE evil (x)",
        "SELECT 1; DROP TABLE users",
    ],
)
def test_unsafe_sql_is_refused_before_execution(executor, tiny_db, sql):
    result = executor.execute(sql)
    assert not result.success
    assert result.error_type == "unsafe_sql"

    # Defense in depth: the table still exists and is untouched.
    connection = sqlite3.connect(tiny_db)
    try:
        count = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    finally:
        connection.close()
    assert count == 3


def test_layer2_authorizer_denies_writes_independently(tiny_db):
    connection = sqlite3.connect(_read_only_uri(tiny_db), uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.set_authorizer(_authorizer)
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            connection.execute("DELETE FROM users")
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            connection.execute("PRAGMA writable_schema = ON")
    finally:
        connection.close()


def test_recursive_cte_timeout_terminates_worker(tiny_db):
    executor = SafeSQLExecutor(tiny_db, max_rows=100, max_execution_ms=250)
    started = time.monotonic()
    result = executor.execute(
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT sum(x) FROM c"
    )
    elapsed = time.monotonic() - started
    assert not result.success
    assert result.timed_out or result.error_type == "timeout"
    assert elapsed < 8.0  # process was terminated, not left running


def test_max_rows_truncates(tiny_db):
    connection = sqlite3.connect(tiny_db)
    with connection:
        connection.execute("CREATE TABLE nums (n INTEGER)")
        connection.executemany("INSERT INTO nums VALUES (?)", [(i,) for i in range(10)])
    connection.close()

    executor = SafeSQLExecutor(tiny_db, max_rows=4, max_execution_ms=1000)
    result = executor.execute("SELECT n FROM nums ORDER BY n")
    assert result.success
    assert result.row_count == 4
    assert result.truncated is True


def test_missing_database_is_schema_error(tmp_path):
    executor = SafeSQLExecutor(tmp_path / "missing.db")
    result = executor.execute("SELECT 1")
    assert not result.success
    assert result.error_type == "schema_error"


def test_sqlite_schema_error_is_reported(executor):
    result = executor.execute("SELECT nope FROM users")
    assert not result.success
    assert result.error_type == "sqlite_error"
    assert "no such column" in result.error_message


def test_validation_errors_on_bad_limits(tiny_db):
    with pytest.raises(ValueError):
        SafeSQLExecutor(tiny_db, max_rows=0)
    with pytest.raises(ValueError):
        SafeSQLExecutor(tiny_db, max_execution_ms=0)


def test_comments_do_not_bypass_single_statement_rule(executor):
    result = executor.execute("SELECT name FROM users /* ; DROP TABLE users */ ORDER BY id")
    assert result.success
    assert result.rows == [["Alice"], ["Bob"], ["Carol"]]
