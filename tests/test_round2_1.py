"""Round 2.1 pre-training gate tests.

These tests cover the formal-config, provenance, paired-ablation, split
fail-closed, crash-resume, reward-trace, artifact-isolation, and config-loader
fixes requested for the PRE-TRAINING GATE review. They are CPU-only and do not
download BIRD or Teacher 4B.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from querydistill.data.audit import audit_data
from querydistill.data.dataset import build_prompt_rows
from querydistill.data.paired import (
    build_paired_targets,
)
from querydistill.data.schema import DistillationRecord, load_distillation_records
from querydistill.data.split_policy import (
    TrainingSplitViolation,
    assert_calibration_splits,
    assert_training_splits,
)
from querydistill.distillation.backends import MockTeacherBackend, TeacherConfig
from querydistill.distillation.pipeline import (
    DistillationConfig,
    DistillationPipeline,
    compute_run_fingerprint,
)
from querydistill.evaluation.modelspec import ModelSpec, infer_model_spec
from querydistill.outputs.prompting import prompt_protocol_spec
from querydistill.quantization.gptq import build_calibration_dataset
from querydistill.rewards.composite import CompositeReward, GoldResultCache
from querydistill.training.callbacks import RewardSampleLogger
from querydistill.training.grpo_backend import (
    GRPOSmokeConfig,
    GRPOSmokeRunner,
    SQLRewardFunction,
)
from querydistill.training.llamafactory_backend import QLoRAConfig
from querydistill.utils import (
    UnknownConfigFieldError,
    strict_dataclass_from_dict,
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


def _distillation_record(
    example_id: str, candidate_sql: str = "SELECT 1", **overrides
) -> DistillationRecord:
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


class _WordTokenizer:
    def __call__(self, text, add_special_tokens=True, **kwargs):
        return {
            "input_ids": list(range(len(text.split()))),
            "attention_mask": [1] * len(text.split()),
        }


# ---------------------------------------------------------------------------
# P0-1 formal GRPO config
# ---------------------------------------------------------------------------


def test_formal_grpo_config_parses():
    payload = yaml.safe_load((PROJECT_ROOT / "configs/grpo/local.yaml").read_text(encoding="utf-8"))
    config = strict_dataclass_from_dict(GRPOSmokeConfig, payload, source="configs/grpo/local.yaml")
    assert config.base_model_path == "models/qwen3-0.6b-base"
    assert config.init_adapter_path == "checkpoints/sft/distilled_local"
    assert config.init_merged_model_path is None
    assert config.require_plan is False


def test_formal_grpo_config_requires_sft_artifact():
    payload = yaml.safe_load((PROJECT_ROOT / "configs/grpo/local.yaml").read_text(encoding="utf-8"))
    config = strict_dataclass_from_dict(GRPOSmokeConfig, payload, source="configs/grpo/local.yaml")
    assert config.init_adapter_path or config.init_merged_model_path
    problems = GRPOSmokeConfig(base_model_path="x").validate(dry_run=True)
    assert any("initialize from an SFT artifact" in p for p in problems)


def test_formal_protocol_is_sql_only():
    spec = prompt_protocol_spec(include_plan=False)
    assert "<sql>" in spec
    assert "<plan>" not in spec
    assert (
        DistillationConfig(
            examples_path=Path("x"), registry_path=Path("y"), output_path=Path("z")
        ).require_plan
        is False
    )
    assert GRPOSmokeConfig(init_adapter_path="x").require_plan is False
    assert QLoRAConfig().include_plan is False


# ---------------------------------------------------------------------------
# P0-2 formal GPTQ / vLLM artifact chain
# ---------------------------------------------------------------------------


def test_formal_config_artifact_chain():
    gptq_payload = yaml.safe_load(
        (PROJECT_ROOT / "configs/quant/gptq_int4_local.yaml").read_text(encoding="utf-8")
    )
    vllm_payload = yaml.safe_load(
        (PROJECT_ROOT / "configs/serving/vllm_gptq_local.yaml").read_text(encoding="utf-8")
    )
    smoke_vllm_payload = yaml.safe_load(
        (PROJECT_ROOT / "configs/serving/vllm_smoke.yaml").read_text(encoding="utf-8")
    )
    assert gptq_payload["adapter_path"] == "checkpoints/grpo/distilled_grpo_local/adapter"
    assert gptq_payload["merged_output_dir"] == "checkpoints/merged/distilled_grpo_local"
    assert gptq_payload["output_dir"] == "checkpoints/gptq/distilled_grpo_int4"
    assert vllm_payload["model_path"] == "checkpoints/gptq/distilled_grpo_int4"
    assert smoke_vllm_payload["model_path"] == "artifacts/smoke/gptq/quantized"


# ---------------------------------------------------------------------------
# P0-3 real teacher provenance
# ---------------------------------------------------------------------------


def test_real_teacher_provenance_not_mock():
    teacher = TeacherConfig(
        model_id="Qwen/Qwen3-4B",
        revision="main",
        prompt_version="v2",
        temperature=0.2,
        max_new_tokens=128,
    )
    assert teacher.model_id != "mock-teacher-1.0"
    assert teacher.provenance()["model_id"] == "Qwen/Qwen3-4B"
    assert teacher.generation_config()["temperature"] == 0.2
    assert teacher.provenance()["prompt_version"] == "v2"


def test_real_teacher_generation_config_in_fingerprint():
    fp1 = compute_run_fingerprint(
        Path("data/tiny_sql/examples.jsonl"),
        "Qwen/Qwen3-4B",
        "main",
        "v2",
        {"temperature": 0.2, "max_new_tokens": 128},
        4,
    )
    fp2 = compute_run_fingerprint(
        Path("data/tiny_sql/examples.jsonl"),
        "Qwen/Qwen3-4B",
        "main",
        "v2",
        {"temperature": 0.7, "max_new_tokens": 128},
        4,
    )
    assert fp1 != fp2


def test_teacher_model_change_changes_fingerprint():
    examples_path = PROJECT_ROOT / "data/tiny_sql/examples.jsonl"
    fp1 = compute_run_fingerprint(examples_path, "mock", "r", "v1", {}, 1)
    fp2 = compute_run_fingerprint(examples_path, "Qwen/Qwen3-4B", "r", "v1", {}, 1)
    assert fp1["teacher_model"] != fp2["teacher_model"]


def test_teacher_sampling_change_changes_fingerprint():
    examples_path = PROJECT_ROOT / "data/tiny_sql/examples.jsonl"
    fp1 = compute_run_fingerprint(examples_path, "m", "r", "v1", {"temperature": 0.1}, 1)
    fp2 = compute_run_fingerprint(examples_path, "m", "r", "v1", {"temperature": 0.9}, 1)
    assert fp1 != fp2


# ---------------------------------------------------------------------------
# P0-5 paired Gold/Distilled manifest
# ---------------------------------------------------------------------------


def _paired_examples():
    return [
        sample_example(example_id="ex-1", split="train"),
        sample_example(example_id="ex-2", split="train", question="q2"),
        sample_example(example_id="ex-3", split="train", question="q3"),
    ]


def test_paired_manifest_controls_gold_subset(tmp_path):
    examples_path = _write_examples_file(tmp_path / "examples.jsonl", _paired_examples())
    records = [_distillation_record("ex-1", "SELECT 10")]
    paired = build_paired_targets(
        _paired_examples(), records, examples_path=examples_path, require_all=False
    )
    manifest_path = tmp_path / "paired_manifest.json"
    paired.write_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["example_ids"] == ["ex-1"]
    assert manifest["requested_train_example_ids"] == ["ex-1", "ex-2", "ex-3"]
    gold_examples = [e for e in _paired_examples() if e.example_id in manifest["example_ids"]]
    assert [e.example_id for e in gold_examples] == ["ex-1"]


def test_gold_and_distilled_use_same_paired_manifest(tmp_path):
    examples_path = _write_examples_file(tmp_path / "examples.jsonl", _paired_examples())
    records = [
        _distillation_record("ex-1", "SELECT 10"),
        _distillation_record("ex-2", "SELECT 20"),
    ]
    paired = build_paired_targets(
        _paired_examples(), records, examples_path=examples_path, require_all=False
    )
    assert sorted(paired.gold_targets) == sorted(paired.distilled_targets) == paired.example_ids


# ---------------------------------------------------------------------------
# P0-6 coverage math
# ---------------------------------------------------------------------------


def test_verified_teacher_coverage_math(tmp_path):
    examples_path = _write_examples_file(tmp_path / "examples.jsonl", _paired_examples())
    records = [_distillation_record("ex-1", "SELECT 10")]
    paired = build_paired_targets(
        _paired_examples(), records, examples_path=examples_path, require_all=False
    )
    assert paired.requested_count == 3
    assert paired.paired_count == 1
    assert paired.verified_teacher_coverage == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# P0-7 training split cannot be expanded
# ---------------------------------------------------------------------------


def test_sft_config_cannot_enable_test():
    with pytest.raises(TrainingSplitViolation):
        assert_training_splits(["train", "test"], policy_name="sft")


def test_grpo_config_cannot_enable_test(tiny_environment):
    config = GRPOSmokeConfig(
        base_model_path="x",
        init_adapter_path="y",
        allowed_splits=frozenset({"train", "test"}),
        examples_path="nope.jsonl",
        registry_path="nope.json",
    )
    with pytest.raises(TrainingSplitViolation):
        GRPOSmokeRunner(config, environment=tiny_environment)


def test_distillation_config_cannot_enable_test(tmp_path, tiny_db):
    examples_path = _write_examples_file(
        tmp_path / "examples.jsonl",
        [sample_example(example_id="train-1", split="train")],
    )
    registry_path = _registry_file(tmp_path / "db_registry.json", tiny_db)
    config = DistillationConfig(
        examples_path=examples_path,
        registry_path=registry_path,
        output_path=tmp_path / "out.jsonl",
        allowed_splits=frozenset({"train", "test"}),
        dry_run=True,
    )
    with pytest.raises(TrainingSplitViolation):
        DistillationPipeline(config).run()


def test_gptq_calibration_cannot_enable_test():
    with pytest.raises(TrainingSplitViolation):
        assert_calibration_splits({"train", "test"})
    with pytest.raises(TrainingSplitViolation):
        build_calibration_dataset(
            "nope.jsonl",
            _WordTokenizer(),
            allowed_splits=frozenset({"train", "test"}),
        )


# ---------------------------------------------------------------------------
# P0-8 audit-data distillation leakage
# ---------------------------------------------------------------------------


def test_audit_fails_when_distillation_contains_test_data(tmp_path, tiny_db):
    examples = [
        sample_example(example_id="train-1", split="train"),
        sample_example(example_id="test-1", split="test", question="q2"),
    ]
    examples_path = _write_examples_file(tmp_path / "examples.jsonl", examples)
    registry_path = _registry_file(tmp_path / "db_registry.json", tiny_db)
    dist_path = tmp_path / "distilled.jsonl"
    dist_path.write_text(
        json.dumps(_distillation_record("test-1", "SELECT 1").model_dump()) + "\n",
        encoding="utf-8",
    )
    report = audit_data(examples_path, registry_path, dist_path)
    assert report["distillation"] is not None
    assert report["ok"] is False


# ---------------------------------------------------------------------------
# P1-1 crash-resume
# ---------------------------------------------------------------------------


class _CrashAfterOneBackend:
    name = "crash-after-one"

    def __init__(self, delegate):
        self.delegate = delegate
        self.calls = 0

    def generate(self, prompt, context=None, num_candidates=1):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("simulated mid-run crash")
        return self.delegate.generate(prompt, context=context, num_candidates=num_candidates)

    def unload(self):
        self.delegate.unload()


def test_distillation_resume_after_real_midrun_crash(tmp_path, tiny_db):
    examples = [
        sample_example(example_id=f"ex-{i}", split="train", question=f"q{i}") for i in range(3)
    ]
    examples_path = _write_examples_file(tmp_path / "examples.jsonl", examples)
    registry_path = _registry_file(tmp_path / "db_registry.json", tiny_db)
    output = tmp_path / "out.jsonl"
    base_kwargs = dict(
        examples_path=examples_path,
        registry_path=registry_path,
        output_path=output,
        teacher_model="mock-teacher-1.0",
        teacher_prompt_version="v1",
        num_candidates=1,
    )
    crash_backend = _CrashAfterOneBackend(
        MockTeacherBackend(
            strategy="gold",
            gold_oracle={e.example_id: e.gold_sql for e in examples},
        )
    )
    first = DistillationPipeline(DistillationConfig(**base_kwargs), backend=crash_backend)
    with pytest.raises(RuntimeError, match="simulated"):
        first.run()
    assert output.exists()
    assert (
        json.loads((output.with_suffix(".manifest.json")).read_text(encoding="utf-8"))["status"]
        == "RUNNING"
    )

    resume = DistillationPipeline(
        DistillationConfig(**base_kwargs, resume=True),
        backend=MockTeacherBackend(
            strategy="gold",
            gold_oracle={e.example_id: e.gold_sql for e in examples},
        ),
    )
    result = resume.run()
    assert result["skipped"] == 1
    assert result["generated"] == 2
    assert len(load_distillation_records(output)) == 3
    manifest = json.loads((output.with_suffix(".manifest.json")).read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETED"
    assert "run_id" in manifest


# ---------------------------------------------------------------------------
# P1-2 distillation gold executes once per example
# ---------------------------------------------------------------------------


def test_distillation_gold_executes_once_per_example(tmp_path, tiny_db, monkeypatch):
    example = sample_example(example_id="ex-1", split="train")
    examples_path = _write_examples_file(tmp_path / "examples.jsonl", [example])
    registry_path = _registry_file(tmp_path / "db_registry.json", tiny_db)
    from querydistill.sql.environment import SQLExecutionEnvironment

    env = SQLExecutionEnvironment.from_registry(registry_path)
    gold_executions = 0
    original_execute = env.execute

    def counting_execute(db_id, sql):
        nonlocal gold_executions
        if sql.strip() == example.gold_sql.strip():
            gold_executions += 1
        return original_execute(db_id, sql)

    monkeypatch.setattr(env, "execute", counting_execute)
    output = tmp_path / "out.jsonl"
    pipeline = DistillationPipeline(
        DistillationConfig(
            examples_path=examples_path,
            registry_path=registry_path,
            output_path=output,
            num_candidates=4,
            backend_name="mock",
            backend_kwargs={
                "strategy": "constant",
                "constant_sql": "SELECT name FROM users ORDER BY id",
            },
        ),
        environment=env,
    )
    pipeline.run()
    assert gold_executions == 1


# ---------------------------------------------------------------------------
# P1-3 gold cache database fingerprint
# ---------------------------------------------------------------------------


def test_gold_cache_invalidates_when_database_changes(tmp_path, tiny_environment, tiny_example):
    cache_dir = tmp_path / "cache"
    gold = tiny_environment.execute("tiny", tiny_example.gold_sql)
    cache_a = GoldResultCache(cache_dir, "hash", database_fingerprints={"tiny": "db-aaa"})
    cache_b = GoldResultCache(cache_dir, "hash", database_fingerprints={"tiny": "db-bbb"})
    cache_a.put("tiny", "ex-001", tiny_example.gold_sql, gold)
    assert cache_a.get("tiny", "ex-001", tiny_example.gold_sql) is not None
    assert cache_b.get("tiny", "ex-001", tiny_example.gold_sql) is None


# ---------------------------------------------------------------------------
# P1-4 reward trace bounded / cleared
# ---------------------------------------------------------------------------


def test_reward_trace_buffer_is_bounded_or_cleared(tmp_path, tiny_environment, tiny_example):
    rows, registry = build_prompt_rows([tiny_example])
    function = SQLRewardFunction(CompositeReward(tiny_environment), registry)
    function(
        completions=["<sql>SELECT name FROM users</sql>"],
        prompts=[rows[0]["prompt"]],
        example_id=["ex-001"],
        db_id=["tiny"],
    )
    assert len(function.traces) == 1
    function(
        completions=["<sql>SELECT name FROM users</sql>"],
        prompts=[rows[0]["prompt"]],
        example_id=["ex-001"],
        db_id=["tiny"],
    )
    assert len(function.traces) == 1  # not accumulated

    # Also verify RewardSampleLogger clears after consuming.
    logger = RewardSampleLogger(function, tmp_path / "reward_clear.jsonl", registry)
    logger(
        completions=["<sql>SELECT name FROM users</sql>"],
        prompts=[rows[0]["prompt"]],
        example_id=["ex-001"],
        db_id=["tiny"],
    )
    assert len(function.traces) == 0


# ---------------------------------------------------------------------------
# P1-5 run artifact isolation
# ---------------------------------------------------------------------------


def test_new_run_refuses_nonempty_output(tmp_path, tiny_db, tiny_environment):
    examples_path = _write_examples_file(
        tmp_path / "examples.jsonl", [sample_example(example_id="train-1", split="train")]
    )
    registry_path = _registry_file(tmp_path / "db_registry.json", tiny_db)
    out = tmp_path / "grpo"
    out.mkdir()
    (out / "stale.txt").write_text("old", encoding="utf-8")
    config = GRPOSmokeConfig(
        base_model_path="models/qwen3-0.6b-base",
        init_adapter_path="checkpoints/sft/distilled_smoke",
        examples_path=str(examples_path),
        registry_path=str(registry_path),
        output_dir=str(out),
        dry_run=True,
    )
    with pytest.raises(FileExistsError):
        GRPOSmokeRunner(config, environment=tiny_environment).run(dry_run=True)


def test_resume_requires_matching_run_identity(tmp_path, tiny_db, tiny_environment):
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
# P1-6 CLI benchmark api-key propagation
# ---------------------------------------------------------------------------


def test_cli_benchmark_propagates_api_key(monkeypatch):
    from typer.testing import CliRunner

    from querydistill.cli import app

    captured: dict = {}

    def fake_benchmark(prompts, endpoint, model, concurrency, max_tokens, api_key):
        captured["api_key"] = api_key
        return {"ok": True}

    monkeypatch.setattr("querydistill.serving.vllm.benchmark", fake_benchmark)
    result = CliRunner().invoke(app, ["benchmark", "--prompt", "p", "--api-key", "secret"])
    assert result.exit_code == 0
    assert captured["api_key"] == "secret"


# ---------------------------------------------------------------------------
# P1-7 inference README matches current status
# ---------------------------------------------------------------------------


def test_inference_readme_matches_current_status(tmp_path):
    script_path = PROJECT_ROOT / "scripts/run_inference_smoke.py"
    spec = importlib.util.spec_from_file_location("run_inference_smoke", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path
    payload = {"status": "PASS", "generation_latency_ms": 12.3}
    module.atomic_write_text_md(out, payload)
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "STATUS: PASS" in readme
    assert "generation_latency_ms: 12.3" in readme


# ---------------------------------------------------------------------------
# P1-8 GRPO global_step
# ---------------------------------------------------------------------------


def test_grpo_steps_use_global_step():
    runner = object.__new__(GRPOSmokeRunner)
    runner.run_id = "run-1"
    runner.output_dir = Path("out")
    evidence = runner._learning_signal([{}, {}, {}], {}, "fp-after", 0, global_step=2)
    assert evidence["steps"] == 2


# ---------------------------------------------------------------------------
# P1-9 GPU smoke orchestrator failure summary
# ---------------------------------------------------------------------------


def test_gpu_smoke_script_has_mandatory_failure_summary():
    text = (PROJECT_ROOT / "scripts/gpu_smoke.sh").read_text(encoding="utf-8")
    assert "MANDATORY_FAILURES" in text
    assert "overall_status.json" in text
    assert "exit 1" in text


# ---------------------------------------------------------------------------
# P1-10 ModelSpec
# ---------------------------------------------------------------------------


def test_model_spec_identity():
    spec = ModelSpec(
        stage="gptq",
        base_model_path="models/qwen3-0.6b-base",
        quantized_model_path="checkpoints/gptq/distilled_grpo_int4",
        quantization={"bits": 4},
        hash="abc",
    )
    identity = spec.identity()
    assert identity["stage"] == "gptq"
    assert identity["quantized_model"] == "checkpoints/gptq/distilled_grpo_int4"
    assert identity["quantization"]["bits"] == 4
    inferred = infer_model_spec("some/path", stage="merged")
    assert inferred.merged_model_path == str(Path("some/path").resolve())


# ---------------------------------------------------------------------------
# P2-1 scripts cleanup
# ---------------------------------------------------------------------------


def test_scripts_directory_is_clean():
    scripts = {p.name for p in (PROJECT_ROOT / "scripts").glob("*") if p.is_file()}
    assert scripts == {
        "benchmark_vllm.py",
        "gpu_smoke.sh",
        "package_release.py",
        "run_cpu_checks.sh",
        "run_inference_smoke.py",
        "serve_vllm.sh",
    }


# ---------------------------------------------------------------------------
# P2-2 strict config loader
# ---------------------------------------------------------------------------


def test_strict_config_loader_rejects_unknown_fields():
    with pytest.raises(UnknownConfigFieldError):
        strict_dataclass_from_dict(
            GRPOSmokeConfig, {"model_id": "Qwen/Qwen3-0.6B-Base"}, source="bad.yaml"
        )
    with pytest.raises(UnknownConfigFieldError):
        strict_dataclass_from_dict(
            QLoRAConfig, {"model_name_or_path": "x", "unknown_key": 1}, source="bad.yaml"
        )
