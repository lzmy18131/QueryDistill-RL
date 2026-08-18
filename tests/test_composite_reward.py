"""CompositeReward tests: end-to-end real SQLite scoring."""

from __future__ import annotations

from querydistill.rewards.composite import CompositeReward

CORRECT = "<plan>\ntables: users\n</plan>\n<sql>\nSELECT name FROM users ORDER BY id\n</sql>"
WRONG_BUT_PRETTY = "<plan>\ntables: users\n</plan>\n<sql>\nSELECT 1 AS x\n</sql>"
UNSAFE = "<plan>\ntables: users\n</plan>\n<sql>\nDROP TABLE users\n</sql>"
MALFORMED = "I refuse to answer"
EMPTY_HACK = "<plan>\ntables: users\n</plan>\n<sql>\nSELECT name FROM users WHERE 1 = 0\n</sql>"


def test_correct_sql_gets_high_total(tiny_environment, tiny_example):
    composite = CompositeReward(tiny_environment, require_plan=True)
    breakdown = composite.evaluate(tiny_example, CORRECT)
    assert breakdown.format == 0.05
    assert breakdown.safety == 0.05
    assert breakdown.execution == 0.1
    assert breakdown.correctness == 1.0
    assert breakdown.total == 1.25


def test_wrong_but_pretty_cannot_score_high(tiny_environment, tiny_example):
    composite = CompositeReward(tiny_environment, require_plan=True)
    breakdown = composite.evaluate(tiny_example, WRONG_BUT_PRETTY)
    assert breakdown.total <= 0.25
    assert breakdown.correctness == 0.0


def test_unsafe_sql_hard_minus_one_and_never_executed(tiny_environment, tiny_example):
    composite = CompositeReward(tiny_environment, require_plan=True)
    breakdown = composite.evaluate(tiny_example, UNSAFE)
    assert breakdown.total == -1.0
    assert breakdown.safety == -1.0
    assert breakdown.correctness == 0.0
    assert "not executed" in breakdown.notes["execution_reason"]


def test_malformed_output_is_negative(tiny_environment, tiny_example):
    composite = CompositeReward(tiny_environment, require_plan=True)
    breakdown = composite.evaluate(tiny_example, MALFORMED)
    assert breakdown.total < 0
    assert breakdown.format == -0.4


def test_trace_contains_all_layers(tiny_environment, tiny_example):
    composite = CompositeReward(tiny_environment, require_plan=True)
    trace = composite.evaluate_trace(tiny_example, CORRECT)
    assert trace["reward_breakdown"]["correctness"] == 1.0
    assert trace["safety"]["safe"] is True
    assert trace["candidate_execution"]["success"] is True
    assert trace["verification"]["equivalent"] is True


def test_empty_result_hack_gets_reduced_correctness(tiny_environment, tiny_example):
    composite = CompositeReward(tiny_environment, require_plan=True)
    example = tiny_example.model_copy(update={"gold_sql": "SELECT name FROM users WHERE age > 100"})
    breakdown = composite.evaluate(example, EMPTY_HACK)
    # Gold is genuinely empty; the WHERE-1=0 candidate passes structural sanity
    # but is capped at reduced empty_structural confidence.
    assert breakdown.correctness == 0.25
    assert breakdown.total < 1.0
