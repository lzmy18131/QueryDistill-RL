"""Round 2.2 final pre-training fix tests.

Covers formal Distilled-SFT non-strict coverage, GRPO resume semantics,
ModelLoader stages, distillation resume without redundant teacher calls,
compact reward logging, and release-tree consistency helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from querydistill.artifacts.manifest import config_hash
from querydistill.data.dataset import build_sft_rows
from querydistill.data.paired import build_paired_targets
from querydistill.distillation.backends import MockTeacherBackend
from querydistill.distillation.pipeline import DistillationConfig, DistillationPipeline
from querydistill.evaluation.modelspec import ModelSpec, load_model
from querydistill.rewards.composite import CompositeReward
from querydistill.training.callbacks import RewardSampleLogger
from querydistill.training.grpo_backend import (
    GRPOSmokeConfig,
    GRPOSmokeRunner,
    _training_config_payload,
)
from tests.helpers import sample_example

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_examples_file(path: Path, examples) -> Path:
    path.write_text(
        "\n".join(json.dumps(example.model_dump()) for example in examples) + "\n",
        encoding="utf-8",
    )
    return path


def _registry_file(path: Path, tiny_db: Path) -> Path:
    path.write_text(json.dumps({"databases": {"tiny": str(tiny_db)}}), encoding="utf-8")
    return path


def _distillation_record(example_id: str, candidate_sql: str = "SELECT 1", **overrides):
    from querydistill.data.schema import DistillationRecord

    payload = dict(
        example_id=example_id,
        teacher_model="mock",
        teacher_model_revision="r1",
        teacher_prompt_version="v1",
        candidate_index=0,
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


def _examples_3():
    return [sample_example(example_id=f"ex-{i}", split="train", question=f"q{i}") for i in range(3)]


# ---------------------------------------------------------------------------
# P0-2 formal Distilled-SFT non-100% coverage integration
# ---------------------------------------------------------------------------


def test_formal_distilled_local_has_strict_false():
    payload = yaml.safe_load(
        (PROJECT_ROOT / "configs/sft/distilled_local.yaml").read_text(encoding="utf-8")
    )
    assert payload["strict_distilled"] is False


def test_distilled_local_dataset_build_coverage_two_thirds(tmp_path, tiny_db):
    examples = _examples_3()
    examples_path = _write_examples_file(tmp_path / "examples.jsonl", examples)
    records = [
        _distillation_record("ex-0", "SELECT 10"),
        _distillation_record("ex-1", "SELECT 20"),
    ]
    dist_path = tmp_path / "distilled.jsonl"
    dist_path.write_text(
        "\n".join(json.dumps(r.model_dump()) for r in records) + "\n", encoding="utf-8"
    )
    paired = build_paired_targets(examples, records, examples_path=examples_path, require_all=False)
    assert paired.requested_count == 3
    assert paired.paired_count == 2
    assert paired.verified_teacher_coverage == pytest.approx(2 / 3)
    assert set(paired.distilled_targets) == {"ex-0", "ex-1"}
    # Build SFT rows from the paired subset and assert no gold fallback is used.
    rows = build_sft_rows(
        [e for e in examples if e.example_id in paired.distilled_targets],
        target_sql_by_id=paired.distilled_targets,
    )
    assert {row["example_id"] for row in rows} == {"ex-0", "ex-1"}
    assert all(row["output"].split("\n")[1] in {"SELECT 10", "SELECT 20"} for row in rows)


# ---------------------------------------------------------------------------
# P0-3 GRPO resume semantics
# ---------------------------------------------------------------------------


def test_grpo_resume_flag_does_not_change_training_fingerprint():
    base = GRPOSmokeConfig(
        base_model_path="b",
        init_adapter_path="a",
        examples_path="e",
        registry_path="r",
        output_dir="o",
        resume=False,
        run_id="r1",
    )
    resumed = GRPOSmokeConfig(
        base_model_path="b",
        init_adapter_path="a",
        examples_path="e",
        registry_path="r",
        output_dir="o",
        resume=True,
        run_id="r1",
    )
    assert config_hash(_training_config_payload(base)) == config_hash(
        _training_config_payload(resumed)
    )


def test_grpo_resume_preserves_original_run_id(tmp_path, tiny_db, tiny_environment):
    examples_path = _write_examples_file(
        tmp_path / "examples.jsonl", [sample_example(example_id="train-1", split="train")]
    )
    registry_path = _registry_file(tmp_path / "db_registry.json", tiny_db)
    out = tmp_path / "grpo"
    config = GRPOSmokeConfig(
        base_model_path="models/qwen3-0.6b-base",
        init_adapter_path="checkpoints/sft/distilled_smoke",
        examples_path=str(examples_path),
        registry_path=str(registry_path),
        output_dir=str(out),
        dry_run=True,
        run_id="original-run-id",
    )
    runner = GRPOSmokeRunner(config, environment=tiny_environment)
    runner.run(dry_run=True)
    identity = json.loads((out / "run_identity.json").read_text(encoding="utf-8"))
    assert identity["run_id"] == "original-run-id"

    resumed = GRPOSmokeConfig(
        base_model_path="models/qwen3-0.6b-base",
        init_adapter_path="checkpoints/sft/distilled_smoke",
        examples_path=str(examples_path),
        registry_path=str(registry_path),
        output_dir=str(out),
        dry_run=True,
        resume=True,
    )
    runner2 = GRPOSmokeRunner(resumed, environment=tiny_environment)
    runner2.run(dry_run=True)
    assert runner2.run_id == "original-run-id"


def test_grpo_resume_passes_latest_checkpoint_to_trainer(tmp_path, tiny_db, tiny_environment):
    examples_path = _write_examples_file(
        tmp_path / "examples.jsonl", [sample_example(example_id="train-1", split="train")]
    )
    registry_path = _registry_file(tmp_path / "db_registry.json", tiny_db)
    out = tmp_path / "grpo"
    (out / "trainer").mkdir(parents=True)
    (out / "trainer" / "checkpoint-1").mkdir()
    (out / "trainer" / "checkpoint-2").mkdir()
    config = GRPOSmokeConfig(
        base_model_path="models/qwen3-0.6b-base",
        init_adapter_path="checkpoints/sft/distilled_smoke",
        examples_path=str(examples_path),
        registry_path=str(registry_path),
        output_dir=str(out),
        dry_run=True,
        resume=True,
    )
    runner = GRPOSmokeRunner(config, environment=tiny_environment)
    assert runner._resume_checkpoint().endswith("checkpoint-2")


def test_grpo_resume_rejects_changed_model_dataset_or_init_artifact(
    tmp_path, tiny_db, tiny_environment
):
    examples_path = _write_examples_file(
        tmp_path / "examples.jsonl", [sample_example(example_id="train-1", split="train")]
    )
    registry_path = _registry_file(tmp_path / "db_registry.json", tiny_db)
    out = tmp_path / "grpo"
    config = GRPOSmokeConfig(
        base_model_path="models/qwen3-0.6b-base",
        init_adapter_path="checkpoints/sft/distilled_smoke",
        examples_path=str(examples_path),
        registry_path=str(registry_path),
        output_dir=str(out),
        dry_run=True,
    )
    GRPOSmokeRunner(config, environment=tiny_environment).run(dry_run=True)
    changed = GRPOSmokeConfig(
        base_model_path="models/qwen3-0.6b-base",
        init_adapter_path="checkpoints/sft/other_smoke",
        examples_path=str(examples_path),
        registry_path=str(registry_path),
        output_dir=str(out),
        dry_run=True,
        resume=True,
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        GRPOSmokeRunner(changed, environment=tiny_environment).run(dry_run=True)


# ---------------------------------------------------------------------------
# P0-4 ModelLoader
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text, return_tensors=None, **kwargs):
        self.calls.append(text)
        return self

    def to(self, device):
        return self

    def decode(self, ids, skip_special_tokens=False):
        return "<sql>SELECT 1</sql>"


class _FakeModel:
    def __init__(self):
        self.generate_calls = 0

    def generate(self, **kwargs):
        self.generate_calls += 1
        return [1, 2, 3]

    def eval(self):
        return self


def test_model_loader_base(monkeypatch):
    calls = {}

    def fake_auto(path, **kwargs):
        calls["path"] = path
        return _FakeModel()

    def fake_tokenizer(path, **kwargs):
        calls["tokenizer_path"] = path
        return _FakeTokenizer()

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", fake_auto)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", fake_tokenizer)
    model, tokenizer = load_model(ModelSpec(stage="base", base_model_path="/tmp/base"))
    assert calls["path"] == "/tmp/base"
    assert model.generate_calls == 0


def test_model_loader_adapter_uses_peft(monkeypatch):
    calls = {}

    def fake_auto(path, **kwargs):
        calls["base"] = path
        return _FakeModel()

    def fake_peft(base, path, **kwargs):
        calls["adapter"] = path
        return base

    def fake_tokenizer(path, **kwargs):
        calls["tokenizer_path"] = path
        return _FakeTokenizer()

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", fake_auto)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", fake_tokenizer)
    monkeypatch.setattr("peft.PeftModel.from_pretrained", fake_peft)
    model, _ = load_model(
        ModelSpec(stage="adapter", base_model_path="/tmp/base", adapter_path="/tmp/adapter")
    )
    assert calls["base"] == "/tmp/base"
    assert calls["adapter"] == "/tmp/adapter"


def test_model_loader_merged(monkeypatch):
    calls = {}

    def fake_auto(path, **kwargs):
        calls["path"] = path
        return _FakeModel()

    def fake_tokenizer(path, **kwargs):
        return _FakeTokenizer()

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", fake_auto)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", fake_tokenizer)
    load_model(ModelSpec(stage="merged", merged_model_path="/tmp/merged"))
    assert calls["path"] == "/tmp/merged"


def test_model_loader_gptq(monkeypatch):
    calls = {}

    def fake_auto(path, **kwargs):
        calls["path"] = path
        calls["quantization_config"] = kwargs.get("quantization_config")
        return _FakeModel()

    def fake_tokenizer(path, **kwargs):
        return _FakeTokenizer()

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", fake_auto)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", fake_tokenizer)
    load_model(
        ModelSpec(
            stage="gptq",
            quantized_model_path="/tmp/gptq",
            quantization={"bits": 4},
        )
    )
    assert calls["path"] == "/tmp/gptq"
    assert calls["quantization_config"] == {"bits": 4}


# ---------------------------------------------------------------------------
# P1-1 Distillation resume no redundant teacher calls / run_id
# ---------------------------------------------------------------------------


class _CountingBackend:
    name = "counting"

    def __init__(self, delegate):
        self.delegate = delegate
        self.calls: list[str] = []

    def generate(self, prompt, context=None, num_candidates=1):
        example_id = (context or {}).get("example_id")
        self.calls.append(example_id)
        return self.delegate.generate(prompt, context=context, num_candidates=num_candidates)

    def unload(self):
        self.delegate.unload()


def test_distillation_resume_does_not_call_teacher_for_completed_examples(tmp_path, tiny_db):
    examples = _examples_3()
    examples_path = _write_examples_file(tmp_path / "examples.jsonl", examples)
    registry_path = _registry_file(tmp_path / "db_registry.json", tiny_db)
    output = tmp_path / "out.jsonl"
    oracle = {e.example_id: e.gold_sql for e in examples}
    delegate = MockTeacherBackend(strategy="gold", gold_oracle=oracle)
    first_backend = _CountingBackend(delegate)
    first = DistillationPipeline(
        DistillationConfig(
            examples_path=examples_path,
            registry_path=registry_path,
            output_path=output,
            teacher_model="mock-teacher-1.0",
            teacher_prompt_version="v1",
            num_candidates=1,
        ),
        backend=first_backend,
    )
    first.run()
    assert len(first_backend.calls) == 3

    resume_backend = _CountingBackend(MockTeacherBackend(strategy="gold", gold_oracle=oracle))
    resume = DistillationPipeline(
        DistillationConfig(
            examples_path=examples_path,
            registry_path=registry_path,
            output_path=output,
            teacher_model="mock-teacher-1.0",
            teacher_prompt_version="v1",
            num_candidates=1,
            resume=True,
        ),
        backend=resume_backend,
    )
    result = resume.run()
    assert result["generated"] == 0
    assert result["skipped"] == 3
    assert resume_backend.calls == []


def test_distillation_resume_preserves_run_id(tmp_path, tiny_db):
    examples = _examples_3()
    examples_path = _write_examples_file(tmp_path / "examples.jsonl", examples)
    registry_path = _registry_file(tmp_path / "db_registry.json", tiny_db)
    output = tmp_path / "out.jsonl"
    oracle = {e.example_id: e.gold_sql for e in examples}
    first = DistillationPipeline(
        DistillationConfig(
            examples_path=examples_path,
            registry_path=registry_path,
            output_path=output,
            teacher_model="mock-teacher-1.0",
            teacher_prompt_version="v1",
            num_candidates=1,
            run_id="fixed-distill-run",
        ),
        backend=MockTeacherBackend(strategy="gold", gold_oracle=oracle),
    )
    first.run()
    manifest_path = output.with_suffix(".manifest.json")
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["run_id"] == "fixed-distill-run"

    resume = DistillationPipeline(
        DistillationConfig(
            examples_path=examples_path,
            registry_path=registry_path,
            output_path=output,
            teacher_model="mock-teacher-1.0",
            teacher_prompt_version="v1",
            num_candidates=1,
            resume=True,
        ),
        backend=MockTeacherBackend(strategy="gold", gold_oracle=oracle),
    )
    resume.run()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["run_id"] == "fixed-distill-run"


# ---------------------------------------------------------------------------
# P1-2 compact reward logging
# ---------------------------------------------------------------------------


def _make_logger(tmp_path, tiny_environment, tiny_example, debug_full_trace=False):
    from querydistill.data.dataset import build_prompt_rows

    rows, registry = build_prompt_rows([tiny_example])
    composite = CompositeReward(tiny_environment)
    from querydistill.training.grpo_backend import SQLRewardFunction

    function = SQLRewardFunction(composite, registry)
    logger = RewardSampleLogger(
        function,
        tmp_path / "reward.jsonl",
        registry,
        debug_full_trace=debug_full_trace,
        run_id="run-1",
    )
    return logger, rows, registry


def test_default_reward_log_is_compact(tmp_path, tiny_environment, tiny_example):
    logger, rows, _ = _make_logger(tmp_path, tiny_environment, tiny_example)
    logger(
        completions=["<sql>SELECT name FROM users</sql>"],
        prompts=[rows[0]["prompt"]],
        example_id=["ex-001"],
        db_id=["tiny"],
    )
    record = json.loads((tmp_path / "reward.jsonl").read_text(encoding="utf-8").strip())
    assert "prompt" not in record
    assert "completion" not in record
    assert "trace" not in record
    assert record["run_id"] == "run-1"
    assert record["reward"] is not None
    assert "strict_equivalent" in record
    assert "candidate_row_count" in record


def test_debug_reward_log_can_include_full_trace(tmp_path, tiny_environment, tiny_example):
    logger, rows, _ = _make_logger(tmp_path, tiny_environment, tiny_example, debug_full_trace=True)
    logger(
        completions=["<sql>SELECT name FROM users</sql>"],
        prompts=[rows[0]["prompt"]],
        example_id=["ex-001"],
        db_id=["tiny"],
    )
    record = json.loads((tmp_path / "reward.jsonl").read_text(encoding="utf-8").strip())
    assert "trace" in record
    assert record["trace"] is not None


# ---------------------------------------------------------------------------
# Release tree consistency (lightweight checks)
# ---------------------------------------------------------------------------


def test_scripts_directory_exact_set_again():
    scripts = {p.name for p in (PROJECT_ROOT / "scripts").glob("*") if p.is_file()}
    assert scripts == {
        "benchmark_vllm.py",
        "gpu_smoke.sh",
        "package_release.py",
        "run_cpu_checks.sh",
        "run_inference_smoke.py",
        "serve_vllm.sh",
    }
