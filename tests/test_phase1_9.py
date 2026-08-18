"""Phase 1.9 final pretraining-gate tests.

These tests are self-contained where possible; tests that exercise the full
6601-row BIRD artifacts skip automatically when the raw data is not present
(e.g. a fresh clone without data files).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from querydistill.data.bird import canonical_bird_example_id
from querydistill.data.protocol_lock import (
    is_valid_sha256_hex,
    validate_protocol_lock_hash_fields,
)
from querydistill.data.schema import load_examples
from querydistill.data.split_policy import (
    SplitPolicy,
    TrainingSplitViolation,
    assert_formal_training_source,
    require_explicit_eval_split,
)
from querydistill.sql.environment import SQLExecutionEnvironment
from querydistill.sql.safety import validate_sql
from tests.helpers import sample_example

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data/bird/raw/bird23_train_filtered.jsonl"
SPLITS_DIR = PROJECT_ROOT / "data/bird/splits"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts/final_pretraining_gate"


def _write_registry(tmp_path: Path, db_path: Path) -> Path:
    registry = tmp_path / "db_registry.json"
    registry.write_text(
        json.dumps({"databases": {"tiny": db_path.resolve().as_posix()}}), encoding="utf-8"
    )
    return registry


# ---------------------------------------------------------------------------
# Safety REPLACE semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT REPLACE('abc','a','x')",
        "SELECT REPLACE(name, ',', '') FROM t",
        "WITH x AS (SELECT REPLACE('abc','a','x') AS a) SELECT a FROM x",
    ],
)
def test_scalar_replace_is_safe(sql):
    decision = validate_sql(sql)
    assert decision.safe
    assert decision.error_type == "none"


@pytest.mark.parametrize(
    "sql",
    [
        "REPLACE INTO t(id) VALUES(1)",
        "INSERT OR REPLACE INTO t(id) VALUES(1)",
        "SELECT 1; REPLACE INTO t(id) VALUES(1)",
        "UPDATE t SET a=1",
        "DELETE FROM t",
    ],
)
def test_replace_dml_and_writes_are_unsafe(sql):
    decision = validate_sql(sql)
    assert not decision.safe
    assert decision.error_type == "unsafe_sql"


# ---------------------------------------------------------------------------
# Timeout semantics
# ---------------------------------------------------------------------------


def test_gold_audit_timeout_is_explicit_and_distinct(tmp_path, tiny_db):
    registry = _write_registry(tmp_path, tiny_db)
    env = SQLExecutionEnvironment.from_registry(registry, max_execution_ms=30000)
    assert env.max_execution_ms == 30000
    assert env.executor_for("tiny").max_execution_ms == 30000


def test_grpo_candidate_environment_keeps_candidate_timeout(tmp_path, tiny_db):
    registry = _write_registry(tmp_path, tiny_db)
    env = SQLExecutionEnvironment.from_registry(registry)
    # Candidate/RL execution timeout is intentionally the executor default
    # (3000 ms), not the Gold audit timeout (30000 ms).
    assert env.max_execution_ms == 3000


def test_gold_audit_tool_uses_audit_timeout_constant():
    import importlib.util

    tool_path = PROJECT_ROOT / "tools" / "run_gold_execution_final_audit.py"
    spec = importlib.util.spec_from_file_location("gold_audit_tool", tool_path)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    assert tool.DEFAULT_AUDIT_TIMEOUT_MS == 30000


# ---------------------------------------------------------------------------
# Split policy / formal_validation
# ---------------------------------------------------------------------------


def test_formal_validation_split_is_in_schema():
    example = sample_example(split="formal_validation")
    assert example.split == "formal_validation"


def test_formal_validation_evaluation_allowed():
    assert require_explicit_eval_split("formal_validation") == "formal_validation"


def test_formal_validation_training_rejected():
    with pytest.raises(TrainingSplitViolation):
        SplitPolicy().apply([sample_example(split="formal_validation")])


def test_validation_tuning_training_rejected():
    with pytest.raises(TrainingSplitViolation):
        SplitPolicy().apply([sample_example(split="validation_tuning")])


def test_mini_dev_dev_split_not_selected_by_training_policy():
    examples = [sample_example(example_id="dev-1", split="dev", question="q")]
    selected, report = SplitPolicy().apply(examples)
    assert selected == []
    assert report.excluded_by_split == {"dev": 1}


def test_formal_mode_guard_rejects_non_canonical_source(tmp_path):
    canonical = tmp_path / "formal_train.jsonl"
    wrong = tmp_path / "raw.jsonl"
    examples = [sample_example()]
    with pytest.raises(TrainingSplitViolation):
        assert_formal_training_source(wrong, canonical, examples)


def test_formal_mode_guard_rejects_validation_ids(tmp_path):
    canonical = tmp_path / "formal_train.jsonl"
    examples = [sample_example(example_id="train-1"), sample_example(example_id="val-1")]
    with pytest.raises(TrainingSplitViolation):
        assert_formal_training_source(canonical, canonical, examples, validation_ids={"val-1"})


# ---------------------------------------------------------------------------
# Formal split materialization
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (SPLITS_DIR / "formal_train.jsonl").exists(), reason="formal_train not present"
)
def test_formal_train_jsonl_exists_and_counts_match_manifests():
    train = load_examples(SPLITS_DIR / "formal_train.jsonl", declared_split="train")
    eng = load_examples(
        SPLITS_DIR / "engineering_validation.jsonl", declared_split="validation_tuning"
    )
    fv = load_examples(SPLITS_DIR / "formal_validation.jsonl", declared_split="formal_validation")
    assert len(train) == 6225
    assert len(eng) == 120
    assert len(fv) == 256
    assert len({e.example_id for e in train}) == 6225
    assert len({e.example_id for e in eng}) == 120
    assert len({e.example_id for e in fv}) == 256


@pytest.mark.skipif(
    not (SPLITS_DIR / "formal_train.jsonl").exists(), reason="formal_train not present"
)
def test_three_way_split_union_is_6601_and_disjoint():
    train = load_examples(SPLITS_DIR / "formal_train.jsonl", declared_split="train")
    eng = load_examples(
        SPLITS_DIR / "engineering_validation.jsonl", declared_split="validation_tuning"
    )
    fv = load_examples(SPLITS_DIR / "formal_validation.jsonl", declared_split="formal_validation")
    train_ids = {e.example_id for e in train}
    eng_ids = {e.example_id for e in eng}
    fv_ids = {e.example_id for e in fv}
    assert len(train_ids & eng_ids) == 0
    assert len(train_ids & fv_ids) == 0
    assert len(eng_ids & fv_ids) == 0
    assert len(train_ids | eng_ids | fv_ids) == 6601


@pytest.mark.skipif(not RAW_PATH.exists(), reason="raw BIRD data not present")
def test_full_source_stable_ids_are_unique():
    records = [
        json.loads(line)
        for line in RAW_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [canonical_bird_example_id(r) for r in records]
    assert len(ids) == 6601
    assert len(set(ids)) == 6601


def test_stable_id_cross_process_independent_of_hash_seed(tmp_path):
    record = {
        "db_id": "airline",
        "question": "How many flights?",
        "evidence": "evidence",
        "SQL": "SELECT COUNT(*) FROM Airlines",
    }
    expected = canonical_bird_example_id(record)
    script = (
        "import json,sys;"
        "from querydistill.data.bird import canonical_bird_example_id;"
        "r=json.loads(sys.stdin.read());"
        "print(canonical_bird_example_id(r))"
    )
    outputs = []
    for seed in ("0", "123"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        proc = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps(record),
            text=True,
            capture_output=True,
            env=env,
            cwd=PROJECT_ROOT,
            check=True,
        )
        outputs.append(proc.stdout.strip())
    assert outputs[0] == outputs[1] == expected
    assert "hash" not in expected  # not the legacy built-in hash ID format


@pytest.mark.skipif(
    not (ARTIFACT_DIR / "formal_id_migration.jsonl").exists()
    or not (
        PROJECT_ROOT / "artifacts/formal_readiness/engineering_validation_manifest.json"
    ).exists(),
    reason="migration artifacts not present",
)
def test_historical_engineering_examples_preserved_after_migration():
    migration = [
        json.loads(line)
        for line in (ARTIFACT_DIR / "formal_id_migration.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    migration_by_old_id = {
        m["old_id_if_any"]: m["stable_id"] for m in migration if m["role"] == "validation_tuning"
    }
    old_manifest = json.loads(
        (
            PROJECT_ROOT / "artifacts/formal_readiness/engineering_validation_manifest.json"
        ).read_text(encoding="utf-8")
    )
    new_eng = load_examples(
        SPLITS_DIR / "engineering_validation.jsonl", declared_split="validation_tuning"
    )
    new_ids = {e.example_id for e in new_eng}
    assert len(old_manifest["example_ids"]) == 120
    for old_id in old_manifest["example_ids"]:
        assert old_id in migration_by_old_id
        assert migration_by_old_id[old_id] in new_ids


# ---------------------------------------------------------------------------
# Protocol lock / DB registry
# ---------------------------------------------------------------------------


def test_protocol_lock_hash_validator_accepts_real_hashes():
    lock = {
        "git_commit": "a" * 40,
        "dataset": {"filtered_train_sha256": "b" * 64},
        "splits": {"formal_train_manifest_sha256": "c" * 64},
        "database": {"db_registry_manifest_sha256": "d" * 64, "db_schema_set_sha256": "e" * 64},
        "execution": {"candidate_execution_timeout_ms": 3000},
    }
    assert validate_protocol_lock_hash_fields(lock) == []


def test_protocol_lock_hash_validator_rejects_placeholders():
    lock = {
        "git_commit": "PENDING_FINAL_COMMIT",
        "database": {"db_registry_hash": [], "db_schema_set_sha256": ""},
    }
    errors = validate_protocol_lock_hash_fields(lock)
    assert len(errors) >= 2


@pytest.mark.skipif(
    not (ARTIFACT_DIR / "db_registry_manifest.json").exists(),
    reason="db registry manifest not present",
)
def test_db_registry_airline_hash_is_valid_hex():
    manifest = json.loads((ARTIFACT_DIR / "db_registry_manifest.json").read_text(encoding="utf-8"))
    airline = manifest["databases"]["airline"]
    assert is_valid_sha256_hex(airline["sha256"])
    assert is_valid_sha256_hex(airline["schema_hash"])
    assert airline["sha256"] != "11e57cdf74cb5ba6a4e91940506f08359bee5463fdf2d3bf50a2823c3c4c5e8a"


@pytest.mark.skipif(
    not (ARTIFACT_DIR / "formal_protocol_lock.json").exists(),
    reason="protocol lock not present",
)
def test_final_protocol_lock_hashes_are_valid():
    lock = json.loads((ARTIFACT_DIR / "formal_protocol_lock.json").read_text(encoding="utf-8"))
    assert validate_protocol_lock_hash_fields(lock) == []
    assert lock["database"]["db_registry_manifest_sha256"]  # not list/null
