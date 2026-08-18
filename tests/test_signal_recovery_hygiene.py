"""Phase 1.5 hygiene tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from querydistill.config import Settings
from querydistill.data.schema import Example
from querydistill.data.split_policy import SplitPolicy, TrainingSplitViolation
from querydistill.distillation.pipeline import DistillationConfig, DistillationPipeline
from querydistill.outputs.fallback import extract_fallback_sql
from querydistill.training.grpo_backend import _trainable_param_sha256
from querydistill.training.llamafactory_backend import prepare_dataset_dir


def test_status_report_is_valid_json():
    path = Path("reports/status_report.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "generated_at" in data
    assert "project_root" in data


def test_train_dataset_is_jsonl(tmp_path):
    rows = [{"instruction": "a", "input": "b", "output": "c"}]
    prepare_dataset_dir(rows, tmp_path / "sft", alias="q")
    assert (tmp_path / "sft" / "train.jsonl").exists()
    assert not (tmp_path / "sft" / "train.json").exists()
    info = json.loads((tmp_path / "sft" / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["q"]["file_name"] == "train.jsonl"


def test_validation_tuning_forbidden_in_training():
    examples = [
        Example(
            example_id="v-1",
            db_id="db",
            question="q",
            schema_text="CREATE TABLE t (id int)",
            gold_sql="SELECT 1",
            split="validation_tuning",
            source="s",
            source_version="1",
        )
    ]
    policy = SplitPolicy(allowed_splits={"train"}, policy_name="train_only")
    with pytest.raises(TrainingSplitViolation):
        policy.apply(examples, source_path="x.jsonl")


def _tiny_pipeline(tmp_path, output, **overrides):
    examples = tmp_path / "examples.jsonl"
    rows = [
        {
            "example_id": f"e{i}",
            "db_id": "tiny",
            "question": f"q{i}",
            "schema_text": "CREATE TABLE t (id int)",
            "gold_sql": "SELECT 1",
            "split": "train",
            "source": "s",
            "source_version": "1",
        }
        for i in range(2)
    ]
    examples.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    registry = tmp_path / "db_registry.json"
    registry.write_text(json.dumps({"databases": {"tiny": "tiny.db"}}), encoding="utf-8")
    config = DistillationConfig(
        examples_path=examples,
        registry_path=registry,
        output_path=output,
        backend_name="mock",
        backend_kwargs={"strategy": "gold"},
        **overrides,
    )
    return DistillationPipeline(config)


def test_teacher_progress_units(tmp_path, tiny_db):
    # tiny_db fixture provides a real sqlite file; point registry to it.
    examples = tmp_path / "examples.jsonl"
    rows = [
        {
            "example_id": f"e{i}",
            "db_id": "tiny",
            "question": f"q{i}",
            "schema_text": "CREATE TABLE t (id int)",
            "gold_sql": "SELECT 1",
            "split": "train",
            "source": "s",
            "source_version": "1",
        }
        for i in range(2)
    ]
    examples.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    registry = tmp_path / "db_registry.json"
    registry.write_text(json.dumps({"databases": {"tiny": "tiny.db"}}), encoding="utf-8")
    output = tmp_path / "out.jsonl"
    pipeline = DistillationPipeline(
        DistillationConfig(
            examples_path=examples,
            registry_path=registry,
            output_path=output,
            backend_name="mock",
            backend_kwargs={"strategy": "gold"},
        )
    )
    pipeline.run()
    progress = json.loads(pipeline.progress_path.read_text(encoding="utf-8"))
    assert set(progress) >= {
        "examples_planned",
        "examples_completed",
        "candidates_planned",
        "candidates_completed",
    }


def test_fallback_extractor_accepts_single_sql():
    assert extract_fallback_sql("Here is the SQL: SELECT 1") == "SELECT 1"
    assert extract_fallback_sql("WITH x AS (SELECT 1) SELECT * FROM x") is not None


def test_fallback_extractor_rejects_multiple_sql():
    assert extract_fallback_sql("SELECT 1; DROP TABLE t") is None
    assert extract_fallback_sql("SELECT 1; SELECT 2") is None


def test_trainable_param_sha_only_trainable():
    model = torch.nn.Linear(4, 4)
    # freeze one parameter
    model.weight.requires_grad = False
    snapshot = {
        name: p.detach().float().cpu() for name, p in model.named_parameters() if p.requires_grad
    }
    sha = _trainable_param_sha256(snapshot)
    assert isinstance(sha, str) and len(sha) == 64


def test_drive_tmp_env_policy():
    settings = Settings.load()
    env = settings.child_env()
    assert env["TMPDIR"].startswith("D:\\LLMCache") or env["TMPDIR"].startswith("/mnt/d/LLMCache")
    assert env["TMP"] == env["TMPDIR"]
    assert env["TEMP"] == env["TMPDIR"]
