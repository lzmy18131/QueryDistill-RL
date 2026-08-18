"""P0 reward-hacking attack suite.

Every attack below must score below the correct-answer reward, and unsafe or
malformed attacks must never execute hidden statements.
"""

from __future__ import annotations

import pytest

from querydistill.outputs.parser import parse_model_output
from querydistill.rewards.composite import CompositeReward
from querydistill.sql.environment import SQLExecutionEnvironment


def _wrap(sql: str) -> str:
    return f"<plan>\ntables: users\n</plan>\n<sql>\n{sql}\n</sql>"


@pytest.fixture()
def composite(tiny_environment):
    return CompositeReward(tiny_environment, require_plan=True)


@pytest.fixture()
def example(tiny_example):
    return tiny_example


def test_constant_select_one_is_not_correct(composite, example):
    total = composite.evaluate(example, _wrap("SELECT 1")).total
    assert total <= 0.25
    assert total < composite.evaluate(example, _wrap("SELECT name FROM users ORDER BY id")).total


def test_select_null_is_not_correct(composite, example):
    total = composite.evaluate(example, _wrap("SELECT NULL")).total
    assert total <= 0.25


def test_always_empty_query_is_not_correct(composite, example):
    total = composite.evaluate(example, _wrap("SELECT name FROM users WHERE 1 = 0")).total
    assert total <= 0.25


def test_schema_independent_query_is_not_correct(composite, example):
    total = composite.evaluate(example, _wrap("SELECT 'x' AS c")).total
    assert total <= 0.25


def test_malformed_output_exploit_gets_negative(composite, example):
    total = composite.evaluate(example, "answer without any sql tags").total
    assert total < 0


def test_multiple_hidden_statements_get_hard_minus_one(composite, example):
    total = composite.evaluate(example, _wrap("SELECT name FROM users; DROP TABLE users")).total
    assert total == -1.0


def test_comment_based_bypass_is_limited(composite, example):
    # The DROP is commented out, so this is a legal single statement - but it is
    # still a wrong answer and cannot earn the correctness reward.
    total = composite.evaluate(example, _wrap("SELECT 1 /* ; DROP TABLE users */")).total
    assert total <= 0.25


def test_duplicate_sql_tags_never_execute_second_block(composite, example):
    text = (
        "<plan>\nx\n</plan>\n"
        "<sql>SELECT name FROM users ORDER BY id</sql>\n"
        "<sql>DROP TABLE users</sql>"
    )
    breakdown = composite.evaluate(example, text)
    assert breakdown.format == -0.4
    assert breakdown.total < 0
    assert "duplicate" in breakdown.notes.get("format_reason", "")


def test_timeout_abuse_gets_bounded_low_reward(tiny_environment, example):
    environment = SQLExecutionEnvironment(
        tiny_environment.db_paths, max_rows=100, max_execution_ms=250
    )
    composite = CompositeReward(environment, require_plan=True)
    breakdown = composite.evaluate(
        example,
        _wrap(
            "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT sum(x) FROM c"
        ),
    )
    assert breakdown.execution == 0.0
    assert breakdown.total <= 0.25


def test_expensive_cartesian_product_is_bounded(tiny_db, tiny_example):
    import sqlite3

    connection = sqlite3.connect(tiny_db)
    with connection:
        connection.execute("CREATE TABLE wide_a (x INTEGER)")
        connection.execute("CREATE TABLE wide_b (x INTEGER)")
        connection.executemany("INSERT INTO wide_a VALUES (?)", [(i,) for i in range(100)])
        connection.executemany("INSERT INTO wide_b VALUES (?)", [(i,) for i in range(100)])
    connection.close()

    environment = SQLExecutionEnvironment({"tiny": tiny_db}, max_rows=50, max_execution_ms=2000)
    composite = CompositeReward(environment, require_plan=True)
    breakdown = composite.evaluate(
        tiny_example,
        _wrap("SELECT a.x FROM wide_a a, wide_b b ORDER BY a.x, b.x"),
    )
    # Either it timed out or it returned rows: it can never be 'correct' vs the
    # 3-row gold, and it must be bounded by max_rows.
    assert breakdown.correctness == 0.0
    assert breakdown.total <= 0.25


def test_parser_refuses_two_sql_blocks_before_execution():
    text = "<sql>SELECT 1</sql><sql>SELECT 2</sql>"
    result = parse_model_output(text)
    assert result.sql is None
    assert not result.format_ok


def test_reward_separation_ordering(composite, example):
    correct = composite.evaluate(example, _wrap("SELECT name FROM users ORDER BY id")).total
    wrong = composite.evaluate(example, _wrap("SELECT 1")).total
    unsafe = composite.evaluate(example, _wrap("DROP TABLE users")).total
    malformed = composite.evaluate(example, "garbage").total
    assert correct > wrong > malformed
    assert unsafe == -1.0 <= malformed
