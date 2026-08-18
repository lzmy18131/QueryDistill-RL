"""Distillation pipeline resume / atomic persistence tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from querydistill.data.schema import load_distillation_records
from querydistill.distillation.backends import MockTeacherBackend
from querydistill.distillation.pipeline import DistillationConfig, DistillationPipeline
from querydistill.sql.environment import SQLExecutionEnvironment


def _examples_path(tmp_path: Path, count: int = 3) -> Path:
    path = tmp_path / "examples.jsonl"
    rows = []
    for i in range(count):
        rows.append(
            {
                "example_id": f"ex-{i}",
                "db_id": "tiny",
                "question": f"Question number {i}",
                "schema_text": "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);",
                "gold_sql": "SELECT name FROM users ORDER BY id",
                "split": "train",
                "source": "test",
                "source_version": "1.0",
            }
        )
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _registry_path(tmp_path: Path, db_path: Path) -> Path:
    registry = tmp_path / "db_registry.json"
    registry.write_text(json.dumps({"databases": {"tiny": "tiny.db"}}), encoding="utf-8")
    return registry


def _pipeline(tmp_path, examples_path, registry_path, output, **overrides):
    config = DistillationConfig(
        examples_path=examples_path,
        registry_path=registry_path,
        output_path=output,
        teacher_model="mock-teacher-1.0",
        teacher_prompt_version="v1",
        num_candidates=1,
        backend_name="mock",
        backend_kwargs={"strategy": "gold"},
        **overrides,
    )
    return DistillationPipeline(config)


def test_dry_run_writes_nothing(tmp_path, tiny_db):
    examples = _examples_path(tmp_path)
    registry = _registry_path(tmp_path, tiny_db)
    output = tmp_path / "out.jsonl"
    pipeline = _pipeline(tmp_path, examples, registry, output, dry_run=True)
    result = pipeline.run()
    assert result["dry_run"] is True
    assert result["to_generate"] == 3
    assert not output.exists()


def test_full_mock_run_verifies_candidates(tmp_path, tiny_db):
    examples = _examples_path(tmp_path)
    registry = _registry_path(tmp_path, tiny_db)
    output = tmp_path / "out.jsonl"
    pipeline = _pipeline(tmp_path, examples, registry, output)
    result = pipeline.run()
    assert result["generated"] == 3
    records = load_distillation_records(output)
    assert len(records) == 3
    assert all(record.parse_valid and record.safe for record in records)
    assert all(record.execution_success and record.execution_equivalent for record in records)
    assert pipeline.progress_path.exists()
    progress = json.loads(pipeline.progress_path.read_text(encoding="utf-8"))
    assert progress["candidates_completed"] == 3
    assert progress["examples_completed"] == 3
    assert progress["candidates_planned"] == 3
    assert progress["examples_planned"] == 3


def test_resume_skips_completed_and_finishes_rest(tmp_path, tiny_db):
    examples = _examples_path(tmp_path)
    registry = _registry_path(tmp_path, tiny_db)
    output = tmp_path / "out.jsonl"

    first = _pipeline(tmp_path, examples, registry, output, max_samples=2)
    assert first.run()["generated"] == 2

    second = _pipeline(tmp_path, examples, registry, output, max_samples=3, resume=True)
    result = second.run()
    assert result["generated"] == 1
    assert result["skipped"] == 2
    assert len(load_distillation_records(output)) == 3


def test_without_resume_existing_output_is_refused(tmp_path, tiny_db):
    examples = _examples_path(tmp_path)
    registry = _registry_path(tmp_path, tiny_db)
    output = tmp_path / "out.jsonl"
    first = _pipeline(tmp_path, examples, registry, output, max_samples=1)
    first.run()
    second = _pipeline(tmp_path, examples, registry, output, max_samples=1, resume=False)
    with pytest.raises(FileExistsError):
        second.run()


def test_constant_mock_candidates_are_verified_false(tmp_path, tiny_db):
    examples = _examples_path(tmp_path)
    registry = _registry_path(tmp_path, tiny_db)
    output = tmp_path / "out.jsonl"
    pipeline = DistillationPipeline(
        DistillationConfig(
            examples_path=examples,
            registry_path=registry,
            output_path=output,
            num_candidates=2,
            max_samples=2,
            backend_name="mock",
            backend_kwargs={"strategy": "constant", "constant_sql": "SELECT 1"},
        )
    )
    result = pipeline.run()
    assert result["generated"] == 4
    records = load_distillation_records(output)
    assert all(not record.execution_equivalent for record in records)
    assert all(
        record.parse_valid and record.safe and record.execution_success for record in records
    )


def test_mock_backend_contract():
    backend = MockTeacherBackend(strategy="gold", gold_oracle={"ex-1": "SELECT 1"})
    out = backend.generate(
        "prompt",
        context={"example_id": "ex-1", "db_id": "tiny", "split": "train"},
        num_candidates=2,
    )
    assert len(out) == 2
    assert "SELECT 1" in out[0] and "<sql>" in out[0]
    backend.unload()
    with pytest.raises(ValueError):
        MockTeacherBackend(strategy="unknown")


def test_environment_never_accepts_model_supplied_paths(tmp_path, tiny_db):
    environment = SQLExecutionEnvironment.from_registry(_registry_path(tmp_path, tiny_db))
    with pytest.raises(ValueError):
        environment.execute("../../etc/passwd", "SELECT 1")
    with pytest.raises(KeyError):
        environment.execute("missing", "SELECT 1")
