"""GRPO backend tests: reward wiring, logging, dry-run, CPU blocking."""

from __future__ import annotations

import json

import pytest

from querydistill.data.dataset import build_prompt_rows
from querydistill.rewards.composite import CompositeReward
from querydistill.training.callbacks import RewardSampleLogger
from querydistill.training.grpo_backend import (
    GRPOBlockedError,
    GRPOSmokeConfig,
    GRPOSmokeRunner,
    SQLRewardFunction,
)


def _write_examples(tmp_path, count=2):
    path = tmp_path / "examples.jsonl"
    rows = [
        {
            "example_id": f"e{i}",
            "db_id": "tiny",
            "question": f"Question {i}",
            "schema_text": "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);",
            "gold_sql": "SELECT name FROM users ORDER BY id",
            "split": "train",
            "source": "test",
            "source_version": "1",
        }
        for i in range(count)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    registry = tmp_path / "db_registry.json"
    registry.write_text(json.dumps({"databases": {"tiny": "tiny.db"}}), encoding="utf-8")
    return path, registry


def test_sql_reward_function_uses_real_sqlite(tiny_environment, tiny_example):
    rows, registry = build_prompt_rows([tiny_example])
    function = SQLRewardFunction(CompositeReward(tiny_environment), registry)
    correct = "<sql>SELECT name FROM users ORDER BY id</sql>"
    wrong = "<sql>SELECT 1</sql>"
    unsafe = "<sql>DROP TABLE users</sql>"
    rewards = function(
        completions=[correct, wrong, unsafe],
        prompts=[rows[0]["prompt"]] * 3,
        example_id=[rows[0]["example_id"]] * 3,
        db_id=[rows[0]["db_id"]] * 3,
    )
    assert rewards[0] > 1.0
    assert 0 < rewards[1] <= 0.25
    assert rewards[2] == -1.0


def test_reward_function_rejects_unknown_metadata(tiny_environment):
    function = SQLRewardFunction(CompositeReward(tiny_environment), {})
    with pytest.raises(KeyError):
        function(
            completions=["SELECT 1"],
            prompts=["unknown prompt"],
            example_id=["missing"],
            db_id=["tiny"],
        )


def test_reward_sample_logger_writes_jsonl(tmp_path, tiny_environment, tiny_example):
    rows, registry = build_prompt_rows([tiny_example])
    function = SQLRewardFunction(CompositeReward(tiny_environment), registry)
    log_path = tmp_path / "reward_samples.jsonl"
    logger = RewardSampleLogger(function, log_path, registry, debug_full_trace=True)
    output = logger(
        completions=["<sql>SELECT name FROM users ORDER BY id</sql>"],
        prompts=[rows[0]["prompt"]],
        example_id=[rows[0]["example_id"]],
        db_id=[rows[0]["db_id"]],
    )
    assert output[0] > 1.0
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["example_id"] == "ex-001"
    assert record["reward"] > 1.0
    assert record["trace"]["verification"]["equivalent"] is True


def test_grpo_config_requires_sft_initialization():
    problems = GRPOSmokeConfig().validate(dry_run=True)
    assert any("initialize from an SFT artifact" in p for p in problems)
    good = GRPOSmokeConfig(
        init_adapter_path="whatever/adapter", init_merged_model_path=None
    ).validate(dry_run=True)
    assert good == []


def test_grpo_config_validation():
    assert GRPOSmokeConfig(max_steps=0, init_adapter_path="x").validate(dry_run=True)
    assert GRPOSmokeConfig(init_adapter_path="x").validate(dry_run=True) == []


def test_grpo_runner_dry_run_writes_artifacts(tmp_path, tiny_db):
    examples_path, registry_path = _write_examples(tmp_path)
    config = GRPOSmokeConfig(
        base_model_path="models/qwen3-0.6b-base",
        init_adapter_path="checkpoints/sft/distilled_smoke",
        examples_path=str(examples_path),
        registry_path=str(registry_path),
        output_dir=str(tmp_path / "grpo"),
        dry_run=True,
        max_samples=2,
    )
    runner = GRPOSmokeRunner(config)
    result = runner.run(dry_run=True)
    assert result["status"] == "DRY_RUN"
    out = tmp_path / "grpo"
    assert json.loads((out / "environment.json").read_text(encoding="utf-8"))["environment"][
        "db_ids"
    ] == ["tiny"]
    assert (out / "resolved_config.yaml").exists()
    assert (out / "status.json").exists()
    assert (out / "README.md").exists()


def test_grpo_runner_blocks_without_cuda_and_records(tmp_path, tiny_db):
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA present; blocking path requires CPU environment")
    examples_path, registry_path = _write_examples(tmp_path)
    init_adapter = tmp_path / "adapter"
    init_adapter.mkdir()
    base_model = tmp_path / "base"
    base_model.mkdir()
    config = GRPOSmokeConfig(
        base_model_path=str(base_model),
        init_adapter_path=str(init_adapter),
        examples_path=str(examples_path),
        registry_path=str(registry_path),
        output_dir=str(tmp_path / "grpo_blocked"),
        dry_run=False,
        max_samples=1,
        max_steps=1,
    )
    runner = GRPOSmokeRunner(config)
    with pytest.raises(GRPOBlockedError):
        runner.run(dry_run=False)
    status = json.loads((tmp_path / "grpo_blocked" / "status.json").read_text(encoding="utf-8"))
    assert "NO_CUDA" in status["status"]
    assert status["trained"] is False
