"""Phase 1.7 pre-full hardening tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from querydistill.rewards.composite import CompositeReward
from querydistill.training.grpo_backend import (
    _reward_signal_stats,
    build_grpo_generation_kwargs,
    build_sql_stopping_criteria,
    evaluate_grpo_gates,
)

ROOT = Path(__file__).resolve().parents[1]


def test_grpo_generation_kwargs_use_sql_close():
    kwargs = build_grpo_generation_kwargs()
    assert kwargs == {}


def test_grpo_stopping_criteria_uses_sql_close():
    class _FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return type("E", (), {"input_ids": [42, 43]})()

    criteria = build_sql_stopping_criteria(_FakeTokenizer(), 8)
    assert len(criteria) == 1
    assert criteria[0].stop_ids == [42, 43]


def test_querydistill_data_not_ignored_by_git():
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    proc = subprocess.run(
        ["git", "check-ignore", "-q", "src/querydistill/data/schema.py"],
        cwd=ROOT,
        capture_output=True,
    )
    assert proc.returncode != 0, "src/querydistill/data/schema.py must not be git-ignored"


def test_reward_signal_stats_semantic_variance(tmp_path):
    samples = tmp_path / "reward_samples.jsonl"
    rows = [
        {
            "generation_group_id": "g1",
            "reward": -0.5,
            "parse_ok": True,
            "safe": True,
            "execution_success": True,
            "strict_equivalent": True,
        },
        {
            "generation_group_id": "g1",
            "reward": 0.5,
            "parse_ok": True,
            "safe": True,
            "execution_success": True,
            "strict_equivalent": False,
        },
    ]
    samples.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    stats = _reward_signal_stats(samples)
    assert stats["generation_group_count"] == 1
    assert stats["nonzero_reward_std_group_count"] == 1
    assert stats["semantic_variance_group_count"] == 1
    assert stats["parse_valid_completion_count"] == 2
    assert stats["execution_success_count"] == 2
    assert stats["strict_correct_count"] == 1
    assert stats["all_rewards_finite"] is True


def test_reward_signal_stats_rejects_all_format_only_variance(tmp_path):
    samples = tmp_path / "reward_samples.jsonl"
    rows = [
        {
            "generation_group_id": "g1",
            "reward": -0.5,
            "parse_ok": False,
            "safe": False,
            "execution_success": False,
            "strict_equivalent": False,
        },
        {
            "generation_group_id": "g1",
            "reward": 0.5,
            "parse_ok": False,
            "safe": False,
            "execution_success": False,
            "strict_equivalent": False,
        },
    ]
    samples.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    stats = _reward_signal_stats(samples)
    assert stats["nonzero_reward_std_group_count"] == 1
    assert stats["semantic_variance_group_count"] == 0


def test_composite_reward_returns_finite_for_junk(tiny_environment):
    from querydistill.data.schema import Example

    composite = CompositeReward(tiny_environment)
    example = Example(
        example_id="junk-1",
        db_id="tiny",
        question="q",
        schema_text="CREATE TABLE t (id int)",
        gold_sql="SELECT 1",
        split="train",
        source="test",
        source_version="1",
    )
    trace = composite.score_once(example, "this is not sql at all")
    assert trace is not None
    assert trace.breakdown.total == trace.breakdown.total  # finite (not NaN)
    assert trace.breakdown.total != float("inf")


def _passing_evidence(**overrides):
    evidence = {
        "generation_group_count": 12,
        "nonzero_reward_std_group_count": 7,
        "nonzero_grad_step_count": 7,
        "trainable_param_sha256_changed": True,
        "parameter_delta_l2": 0.05,
        "changed_parameter_tensor_count": 392,
        "parse_valid_completion_count": 10,
        "execution_success_count": 5,
        "semantic_variance_group_count": 5,
        "all_rewards_finite": True,
    }
    evidence.update(overrides)
    return evidence


def test_strong_gate_inherits_strict_gate():
    gates = evaluate_grpo_gates(_passing_evidence())
    assert gates["strict_grpo_signal_gate_pass"] is True
    assert gates["strong_confirmation_gate_pass"] is True


def test_strong_gate_false_when_rewards_not_finite():
    gates = evaluate_grpo_gates(_passing_evidence(all_rewards_finite=False))
    assert gates["strict_grpo_signal_gate_pass"] is False
    assert gates["strong_confirmation_gate_pass"] is False


def test_strong_gate_false_when_parameter_delta_zero():
    gates = evaluate_grpo_gates(_passing_evidence(parameter_delta_l2=0.0))
    assert gates["strict_grpo_signal_gate_pass"] is False
    assert gates["strong_confirmation_gate_pass"] is False


def test_strong_gate_false_when_sha_unchanged():
    gates = evaluate_grpo_gates(_passing_evidence(trainable_param_sha256_changed=False))
    assert gates["strict_grpo_signal_gate_pass"] is False
    assert gates["strong_confirmation_gate_pass"] is False


def test_grpo_trace_schema_has_protocol_fields(tmp_path):
    from querydistill.training.callbacks import RewardSampleLogger

    class FakeReward:
        traces = []

        def __call__(self, completions, prompts=None, **kwargs):
            return [0.1] * len(completions)

    path = tmp_path / "reward_samples.jsonl"
    logger = RewardSampleLogger(
        FakeReward(),
        path,
        registry={},
        debug_full_trace=True,
        run_id="test",
        step_provider=lambda: 3,
    )
    logger(["<sql>SELECT 1</sql>"], prompts=["p"])
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(rows) == 1
    r = rows[0]
    assert r["optimizer_step"] == 3
    assert r["completion_index"] == 0
    assert r["stop_reason"] == "sql_close"
    assert r["sql_block_count"] == 1
