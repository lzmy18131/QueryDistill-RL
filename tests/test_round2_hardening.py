"""Round-2 hardening regression tests.

These tests cover the correctness/data-integrity invariants that are not
already exercised by the dedicated module test files: split isolation, paired
distilled targets, safe model contexts, reward/gold caching, prompt budgets,
artifact manifests, and vLLM bearer auth.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from querydistill.artifacts.manifest import ArtifactManifest
from querydistill.data.dataset import (
    build_prompt_rows,
    build_sft_rows,
    truncate_prompt_to_token_budget,
)
from querydistill.data.paired import (
    DistilledTargetMissingError,
    build_paired_targets,
    select_verified_candidates,
)
from querydistill.data.schema import DistillationRecord
from querydistill.data.split_policy import (
    CALIBRATION_ALLOWED,
    SplitPolicy,
)
from querydistill.distillation.backends import TransformersTeacherBackend
from querydistill.distillation.pipeline import (
    DistillationConfig,
    DistillationFingerprintMismatchError,
    DistillationPipeline,
)
from querydistill.evaluation.harness import TransformersModelBackend
from querydistill.outputs.context import RealModelContext
from querydistill.rewards.composite import CompositeReward, GoldResultCache
from querydistill.serving.vllm import _one_request
from querydistill.training.grpo_backend import GRPOSmokeConfig, GRPOSmokeRunner
from tests.helpers import sample_example


def _write_examples_file(path: Path, examples) -> Path:
    path.write_text(
        "\n".join(json.dumps(example.model_dump()) for example in examples) + "\n",
        encoding="utf-8",
    )
    return path


def _registry_file(path: Path, tiny_db: Path) -> Path:
    path.write_text(json.dumps({"databases": {"tiny": str(tiny_db)}}), encoding="utf-8")
    return path


def _distillation_record(
    example_id: str, candidate_sql: str, candidate_index: int = 0, **overrides
) -> DistillationRecord:
    payload = dict(
        example_id=example_id,
        teacher_model="mock",
        teacher_model_revision="r1",
        teacher_prompt_version="v1",
        candidate_index=candidate_index,
        raw_candidate_output=f"<sql>{candidate_sql}</sql>",
        candidate_sql=candidate_sql,
        candidate_plan=None,
        parse_valid=True,
        safe=True,
        execution_success=True,
        execution_equivalent=True,
        generation_config={},
        created_at="2026-08-17T00:00:00+00:00",
    )
    payload.update(overrides)
    return DistillationRecord(**payload)


class _WordTokenizer:
    """Deterministic tokenizer surrogate for prompt-budget tests."""

    def __call__(self, text, add_special_tokens=True, **kwargs):
        return {
            "input_ids": list(range(len(text.split()))),
            "attention_mask": [1] * len(text.split()),
        }


# ---------------------------------------------------------------------------
# P0-1 train-only split isolation
# ---------------------------------------------------------------------------


def test_split_policy_train_only_excludes_dev_test():
    examples = [
        sample_example(example_id="train-1", split="train"),
        sample_example(example_id="dev-1", split="dev"),
        sample_example(example_id="test-1", split="test"),
    ]
    selected, report = SplitPolicy().apply(examples)
    assert [example.example_id for example in selected] == ["train-1"]
    assert report.excluded_by_split == {"dev": 1, "test": 1}


def test_calibration_split_policy_excludes_dev_test():
    policy = SplitPolicy(allowed_splits=set(CALIBRATION_ALLOWED), policy_name="calibration")
    examples = [
        sample_example(example_id="train-1", split="train"),
        sample_example(example_id="cal-1", split="calibration"),
        sample_example(example_id="dev-1", split="dev"),
        sample_example(example_id="test-1", split="test"),
    ]
    selected, report = policy.apply(examples)
    assert {example.example_id for example in selected} == {"train-1", "cal-1"}
    assert report.excluded_by_split == {"dev": 1, "test": 1}


def test_distillation_dry_run_split_report_is_train_only(tmp_path, tiny_db):
    examples_path = _write_examples_file(
        tmp_path / "examples.jsonl",
        [
            sample_example(example_id="train-1", split="train"),
            sample_example(example_id="test-1", split="test", question="q2"),
        ],
    )
    registry_path = _registry_file(tmp_path / "db_registry.json", tiny_db)
    pipeline = DistillationPipeline(
        DistillationConfig(
            examples_path=examples_path,
            registry_path=registry_path,
            output_path=tmp_path / "out.jsonl",
            dry_run=True,
        )
    )
    result = pipeline.run()
    assert result["split_report"]["included"] == 1
    assert result["split_report"]["excluded_by_split"] == {"test": 1}


def test_grpo_dry_run_split_report_is_train_only(tmp_path, tiny_db):
    examples_path = _write_examples_file(
        tmp_path / "examples.jsonl",
        [
            sample_example(example_id="train-1", split="train"),
            sample_example(example_id="dev-1", split="dev", question="q2"),
            sample_example(example_id="test-1", split="test", question="q3"),
        ],
    )
    registry_path = _registry_file(tmp_path / "db_registry.json", tiny_db)
    config = GRPOSmokeConfig(
        base_model_path="models/qwen3-0.6b-base",
        init_adapter_path="checkpoints/sft/distilled_smoke",
        examples_path=str(examples_path),
        registry_path=str(registry_path),
        output_dir=str(tmp_path / "grpo"),
        dry_run=True,
        max_samples=10,
    )
    runner = GRPOSmokeRunner(config)
    runner.run(dry_run=True)
    env = json.loads((tmp_path / "grpo" / "environment.json").read_text(encoding="utf-8"))
    assert env["split_report"]["included"] == 1
    assert env["split_report"]["excluded_by_split"] == {"dev": 1, "test": 1}


# ---------------------------------------------------------------------------
# P0-2 DistillationRecord schema / no nested protocol
# ---------------------------------------------------------------------------


def test_distillation_record_stores_parsed_sql_not_raw_output():
    record = _distillation_record(
        "ex-1", "SELECT 1", raw_candidate_output="<plan>p</plan>\n<sql>SELECT 1</sql>"
    )
    assert record.candidate_sql == "SELECT 1"
    assert "<sql>" not in record.candidate_sql
    assert record.raw_candidate_output.startswith("<plan>")


def test_distilled_target_not_nested():
    row = build_sft_rows(
        [sample_example()], target_sql_by_id={"ex-001": "SELECT 2"}, include_plan=True
    )[0]
    assert row["output"].count("<sql>") == 1
    assert row["output"].count("</sql>") == 1
    assert "SELECT 2" in row["output"]


def test_distilled_target_single_sql_block():
    row = build_sft_rows(
        [sample_example()], target_sql_by_id={"ex-001": "SELECT 3"}, include_plan=False
    )[0]
    assert row["output"].count("<sql>") == 1
    assert row["output"].count("</sql>") == 1


# ---------------------------------------------------------------------------
# P0-3 paired distilled targets / no gold fallback
# ---------------------------------------------------------------------------


def test_distilled_missing_target_fails():
    with pytest.raises(DistilledTargetMissingError):
        build_sft_rows([sample_example()], target_sql_by_id={})


def test_paired_gold_distilled_same_ids(tmp_path):
    examples_path = tmp_path / "examples.jsonl"
    _write_examples_file(
        examples_path,
        [
            sample_example(example_id="ex-1", split="train"),
            sample_example(example_id="ex-2", split="train", question="q2"),
        ],
    )
    records = [
        _distillation_record("ex-1", "SELECT 10"),
        _distillation_record("ex-2", "SELECT 20"),
    ]
    paired = build_paired_targets(
        [
            sample_example(example_id="ex-1", split="train"),
            sample_example(example_id="ex-2", split="train", question="q2"),
        ],
        records,
        examples_path=examples_path,
        require_all=True,
    )
    assert paired.example_ids == ["ex-1", "ex-2"]
    assert sorted(paired.gold_targets) == sorted(paired.distilled_targets)
    assert set(paired.gold_targets) == {"ex-1", "ex-2"}


def test_verified_candidate_selection_deterministic():
    records = [
        _distillation_record("ex-1", "SELECT second", candidate_index=1),
        _distillation_record("ex-1", "SELECT first", candidate_index=0),
        _distillation_record("ex-1", "SELECT third", candidate_index=2),
    ]
    targets = select_verified_candidates(records, policy="min_candidate_index")
    assert targets["ex-1"] == "SELECT first"


# ---------------------------------------------------------------------------
# P0-7 gold never enters real model contexts
# ---------------------------------------------------------------------------


def test_real_model_context_has_no_gold():
    context = RealModelContext(example_id="e1", db_id="tiny", split="train")
    assert context.as_dict() == {"example_id": "e1", "db_id": "tiny", "split": "train"}
    assert "gold_sql" not in context.as_dict()
    assert RealModelContext.safe_keys() == {"example_id", "db_id", "split"}


def test_teacher_backend_cannot_receive_gold():
    backend = TransformersTeacherBackend(model_id="dummy-model")
    with pytest.raises(ValueError, match="unsafe context"):
        backend.generate("prompt", context={"gold_sql": "SELECT 1"})


def test_evaluation_backend_cannot_receive_gold():
    backend = TransformersModelBackend("dummy-model")
    with pytest.raises(ValueError, match="unsafe context"):
        backend.generate("prompt", context={"gold_sql": "SELECT 1"})


# ---------------------------------------------------------------------------
# P1-2 distillation resume fingerprint
# ---------------------------------------------------------------------------


def test_distillation_resume_rejects_model_change(tmp_path, tiny_db):
    examples_path = _write_examples_file(
        tmp_path / "examples.jsonl", [sample_example(example_id="ex-1", split="train")]
    )
    registry_path = _registry_file(tmp_path / "db_registry.json", tiny_db)
    output = tmp_path / "out.jsonl"
    base_config = DistillationConfig(
        examples_path=examples_path,
        registry_path=registry_path,
        output_path=output,
        teacher_model="mock-teacher-1.0",
        teacher_prompt_version="v1",
        num_candidates=1,
        backend_name="mock",
        backend_kwargs={"strategy": "gold"},
    )
    DistillationPipeline(base_config).run()

    changed = DistillationConfig(
        examples_path=examples_path,
        registry_path=registry_path,
        output_path=output,
        teacher_model="different-teacher",
        teacher_prompt_version="v1",
        num_candidates=1,
        resume=True,
        backend_name="mock",
        backend_kwargs={"strategy": "gold"},
    )
    with pytest.raises(DistillationFingerprintMismatchError):
        DistillationPipeline(changed).run()


def test_distillation_resume_accepts_same_fingerprint(tmp_path, tiny_db):
    examples_path = _write_examples_file(
        tmp_path / "examples.jsonl", [sample_example(example_id="ex-1", split="train")]
    )
    registry_path = _registry_file(tmp_path / "db_registry.json", tiny_db)
    output = tmp_path / "out.jsonl"
    config = DistillationConfig(
        examples_path=examples_path,
        registry_path=registry_path,
        output_path=output,
        teacher_model="mock-teacher-1.0",
        teacher_prompt_version="v1",
        num_candidates=1,
        backend_name="mock",
        backend_kwargs={"strategy": "gold"},
    )
    assert DistillationPipeline(config).run()["generated"] == 1
    resumed = DistillationConfig(
        examples_path=examples_path,
        registry_path=registry_path,
        output_path=output,
        teacher_model="mock-teacher-1.0",
        teacher_prompt_version="v1",
        num_candidates=1,
        resume=True,
        backend_name="mock",
        backend_kwargs={"strategy": "gold"},
    )
    result = DistillationPipeline(resumed).run()
    assert result["skipped"] == 1


# ---------------------------------------------------------------------------
# P1-3 reward / gold cache and single candidate execution
# ---------------------------------------------------------------------------


def test_candidate_executes_once_per_reward(tiny_environment, tiny_example):
    composite = CompositeReward(tiny_environment)
    trace = composite.score_once(tiny_example, "<sql>SELECT name FROM users</sql>")
    assert trace.candidate_executions == 1
    assert composite.execution_count == 1


def test_gold_result_cache_roundtrip(tmp_path, tiny_environment, tiny_example):
    cache = GoldResultCache(tmp_path / "cache", dataset_hash="abc123")
    assert cache.get("tiny", "ex-001", tiny_example.gold_sql) is None
    gold = tiny_environment.execute("tiny", tiny_example.gold_sql)
    cache.put("tiny", "ex-001", tiny_example.gold_sql, gold)
    cached = cache.get("tiny", "ex-001", tiny_example.gold_sql)
    assert cached is not None
    assert cached.success is True
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1


def test_gold_cache_invalidates_on_dataset_change(tmp_path, tiny_environment, tiny_example):
    cache_dir = tmp_path / "cache"
    cache_a = GoldResultCache(cache_dir, dataset_hash="hash-a")
    cache_b = GoldResultCache(cache_dir, dataset_hash="hash-b")
    gold = tiny_environment.execute("tiny", tiny_example.gold_sql)
    cache_a.put("tiny", "ex-001", tiny_example.gold_sql, gold)
    assert cache_b.get("tiny", "ex-001", tiny_example.gold_sql) is None


# ---------------------------------------------------------------------------
# P1-5 duplicate prompts / P1-6 config honored / prompt budget
# ---------------------------------------------------------------------------


def test_duplicate_prompt_examples_do_not_overwrite():
    examples = [
        sample_example(example_id="dup-a", split="train"),
        sample_example(example_id="dup-b", split="train"),
    ]
    rows, registry = build_prompt_rows(examples)
    assert len(rows) == 2
    assert rows[0]["prompt"] == rows[1]["prompt"]
    assert registry["dup-a"].example_id == "dup-a"
    assert registry["dup-b"].example_id == "dup-b"


def test_grpo_config_honored_in_dry_run(tmp_path, tiny_db):
    examples_path = _write_examples_file(
        tmp_path / "examples.jsonl", [sample_example(example_id="train-1", split="train")]
    )
    registry_path = _registry_file(tmp_path / "db_registry.json", tiny_db)
    config = GRPOSmokeConfig(
        base_model_path="models/qwen3-0.6b-base",
        init_adapter_path="checkpoints/sft/distilled_smoke",
        examples_path=str(examples_path),
        registry_path=str(registry_path),
        output_dir=str(tmp_path / "grpo"),
        dry_run=True,
        max_prompt_length=123,
        max_completion_length=64,
        use_gradient_checkpointing=True,
        use_vllm=True,
        lora_target_modules=["q_proj", "v_proj"],
    )
    GRPOSmokeRunner(config).run(dry_run=True)
    env = json.loads((tmp_path / "grpo" / "environment.json").read_text(encoding="utf-8"))
    resolved = env["config"]
    assert resolved["max_prompt_length"] == 123
    assert resolved["max_completion_length"] == 64
    assert resolved["use_gradient_checkpointing"] is True
    assert resolved["use_vllm"] is True
    assert resolved["lora_target_modules"] == ["q_proj", "v_proj"]


def test_prompt_budget_enforced():
    tokenizer = _WordTokenizer()
    schema = "\n".join(f"CREATE TABLE t{i} (c{i} INTEGER);" for i in range(30))
    truncated, flagged = truncate_prompt_to_token_budget(
        tokenizer,
        question="List everything?",
        schema_text=schema,
        db_id="db",
        include_plan=False,
        max_prompt_tokens=50,
    )
    assert flagged is True
    assert len(tokenizer(truncated, add_special_tokens=False)["input_ids"]) <= 50


def test_sft_has_no_placeholder_plan():
    row = build_sft_rows([sample_example()], include_plan=True)[0]
    assert "tables: from schema" not in row["output"]
    assert "joins: as needed" not in row["output"]
    assert "tables:" in row["output"]


# ---------------------------------------------------------------------------
# P1-10 vLLM bearer auth / P0-5 artifact chain
# ---------------------------------------------------------------------------


def test_vllm_client_sends_bearer_token(monkeypatch):
    captured: dict = {}

    class FakeResponse:
        def iter_lines(self, decode_unicode=False):
            return iter(
                [
                    'data: {"choices":[{"delta":{"content":"x"}}]}',
                    "data: [DONE]",
                ]
            )

        def raise_for_status(self):
            return None

    def fake_post(url, json, headers, stream, timeout):
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)
    _one_request("prompt", "http://127.0.0.1:8000/v1", "model", 10, "secret-key")
    assert captured["headers"]["Authorization"] == "Bearer secret-key"


def test_stage_artifact_chain_manifests(tmp_path):
    stages = [
        ("base", None, "checkpoints/base"),
        ("sft", "checkpoints/base", "checkpoints/sft/distilled"),
        ("grpo", "checkpoints/sft/distilled", "checkpoints/grpo/distilled"),
        ("gptq", "checkpoints/grpo/distilled", "checkpoints/gptq/int4"),
    ]
    for stage, input_artifact, output_artifact in stages:
        manifest = ArtifactManifest(
            stage=stage,
            input_artifact=input_artifact,
            output_artifact=output_artifact,
            base_model="models/qwen3-0.6b-base",
            adapter=output_artifact if stage in {"sft", "grpo"} else None,
            config_hash="abc",
        )
        path = manifest.write(tmp_path / stage)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["stage"] == stage
        assert payload["output_artifact"] == output_artifact
