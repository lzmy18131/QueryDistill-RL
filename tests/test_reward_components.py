"""Individual reward component tests."""

from __future__ import annotations

from querydistill.outputs.parser import parse_model_output
from querydistill.rewards.correctness_reward import correctness_reward
from querydistill.rewards.execution_reward import execution_reward
from querydistill.rewards.format_reward import format_reward
from querydistill.rewards.parse_reward import parse_reward
from querydistill.rewards.safety_reward import safety_reward
from querydistill.sql.executor import ExecutionResult
from querydistill.sql.safety import validate_sql
from querydistill.sql.verifier import VerificationResult


def _verification(equivalent=True, kind="unordered_rows", partial=False):
    return VerificationResult(
        equivalent=equivalent,
        strict_equivalent=equivalent,
        partial_credit=partial,
        kind=kind,
        reason="test",
        candidate_row_count=1,
        gold_row_count=1,
    )


def test_format_reward_valid_and_malformed():
    valid = parse_model_output("<plan>\nx\n</plan>\n<sql>SELECT 1</sql>")
    assert format_reward(valid, require_plan=True)[0] == 0.05
    malformed = parse_model_output("no tags at all")
    assert format_reward(malformed)[0] == -0.4
    fence = parse_model_output("```sql\nSELECT 1\n```")
    assert format_reward(fence)[0] == 0.0


def test_parse_reward_valid_and_invalid():
    valid = parse_model_output("<sql>SELECT 1</sql>")
    assert parse_reward(valid)[0] == 0.05
    invalid = parse_model_output("<sql>SELEC 1</sql>")
    assert parse_reward(invalid)[0] == -0.4
    none = parse_model_output("nothing")
    assert parse_reward(none)[0] == -0.4


def test_safety_reward_ranges():
    assert safety_reward(validate_sql("SELECT 1"))[0] == 0.05
    assert safety_reward(validate_sql("DROP TABLE x"))[0] == -1.0
    assert safety_reward(None)[0] == 0.0


def test_execution_reward_ranges():
    success = ExecutionResult(success=True, rows=[[1]], columns=["x"], row_count=1)
    assert execution_reward(success)[0] == 0.1
    failed = ExecutionResult(success=False, error_type="sqlite_error", error_message="x")
    assert execution_reward(failed)[0] == 0.0
    assert execution_reward(None)[0] == 0.0


def test_correctness_reward_dominance_rules():
    assert correctness_reward(_verification(True, "unordered_rows"))[0] == 1.0
    assert correctness_reward(_verification(True, "ordered_rows"))[0] == 1.0
    assert (
        correctness_reward(_verification(False, "empty_structural_partial", partial=True))[0]
        == 0.25
    )
    assert correctness_reward(_verification(False))[0] == 0.0
    assert correctness_reward(None)[0] == 0.0


def test_correctness_dominates_pretty_wrong_output():
    wrong_pretty_total = 0.05 + 0.05 + 0.05 + 0.1 + correctness_reward(_verification(False))[0]
    correct_minimal_total = 0.05 + 0.05 + 0.05 + 0.1 + correctness_reward(_verification(True))[0]
    assert wrong_pretty_total <= 0.25
    assert correct_minimal_total >= 1.25
    assert correct_minimal_total > wrong_pretty_total
