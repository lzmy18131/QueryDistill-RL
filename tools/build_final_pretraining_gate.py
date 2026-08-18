#!/usr/bin/env python3
"""Build Phase 1.9 final pretraining-gate artifacts.

This script materializes the canonical three-way split files with stable
SHA-256 BIRD example IDs, rebuilds the 80-DB registry manifest with the
resolved airline database, runs the leakage audit, and writes the formal
protocol lock hashes.

It does NOT run the Gold execution audit (see
``tools/run_gold_execution_final_audit.py``); it only prepares the data and
database artifacts that the audit consumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from querydistill.data.bird import (
    bird_content_signature,
    canonical_bird_example_id,
    load_bird_mini_dev,
    load_bird_train_filtered,
    schema_for_db,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = PROJECT_ROOT / "data/bird/raw/bird23_train_filtered.jsonl"
DEFAULT_MINI = PROJECT_ROOT / "data/bird/raw/mini_dev_sqlite.json"
DEFAULT_REGISTRY = PROJECT_ROOT / "data/bird/db_registry.json"
DEFAULT_SPLITS = PROJECT_ROOT / "data/bird/splits"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts/final_pretraining_gate"


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _db_paths_from_registry(registry_path: Path) -> dict[str, Path]:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    raw = payload.get("databases", payload)
    db_paths: dict[str, Path] = {}
    for db_id, rel in raw.items():
        path = Path(str(rel))
        if not path.is_absolute():
            path = registry_path.parent / path
        db_paths[db_id] = path.resolve()
    return db_paths


def _schema_hash(schema_text: str) -> str:
    return sha256_text(schema_text)


def _db_introspection(db_path: Path) -> dict:
    con = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        quick = con.execute("PRAGMA quick_check").fetchone()[0]
        rows = con.execute(
            "SELECT type, name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name"
        ).fetchall()
        table_count = sum(1 for t, _ in rows if t == "table")
        view_count = sum(1 for t, _ in rows if t == "view")
        return {
            "quick_check": str(quick),
            "table_count": table_count,
            "view_count": view_count,
            "table_names": [name for _, name in rows],
        }
    finally:
        con.close()


def _build_db_registry_manifest(registry_path: Path) -> dict:
    db_paths = _db_paths_from_registry(registry_path)
    required_train = sorted(set(r["db_id"] for r in load_bird_train_filtered(DEFAULT_RAW)))
    required_eval = sorted(set(r["db_id"] for r in load_bird_mini_dev(DEFAULT_MINI)))
    all_required = sorted(set(required_train) | set(required_eval))

    databases: dict[str, dict] = {}
    missing: list[str] = []
    read_only_failures: list[str] = []
    integrity_failures: list[str] = []
    introspection_failures: list[str] = []
    schema_missing: list[str] = []
    hash_conflicts: list[dict] = []

    for db_id in all_required:
        path = db_paths.get(db_id)
        entry: dict = {
            "db_id": db_id,
            "logical_path": None,
            "absolute_runtime_path": None,
            "source": "birdsql/bird23-train-filtered official train.zip",
            "source_revision": "official train.zip / mini_dev",
            "file_size_bytes": None,
            "sha256": None,
            "file_exists": False,
            "quick_check": None,
            "read_only_open": None,
            "schema_hash": None,
            "table_count": None,
            "view_count": None,
            "schema_introspection": None,
        }
        if path is None or not path.is_file():
            missing.append(db_id)
            databases[db_id] = entry
            continue
        entry["logical_path"] = path.relative_to(registry_path.parent).as_posix()
        entry["absolute_runtime_path"] = str(path)
        entry["file_size_bytes"] = path.stat().st_size
        entry["sha256"] = sha256_file(path)
        entry["file_exists"] = True
        try:
            con = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
            con.close()
            entry["read_only_open"] = True
        except Exception:  # noqa: BLE001
            entry["read_only_open"] = False
            read_only_failures.append(db_id)
            databases[db_id] = entry
            continue
        info = _db_introspection(path)
        if info["quick_check"] != "ok":
            integrity_failures.append(db_id)
        entry.update(
            {
                "quick_check": info["quick_check"],
                "table_count": info["table_count"],
                "view_count": info["view_count"],
            }
        )
        try:
            schema_text = schema_for_db(path)
        except Exception:  # noqa: BLE001
            introspection_failures.append(db_id)
            entry["schema_introspection"] = False
            databases[db_id] = entry
            continue
        if not schema_text:
            schema_missing.append(db_id)
        entry["schema_introspection"] = True
        entry["schema_hash"] = _schema_hash(schema_text)
        databases[db_id] = entry

    manifest = {
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(),
        "required_train_db_ids": required_train,
        "required_eval_db_ids": required_eval,
        "all_required_db_ids": all_required,
        "train_unique_db_count": len(required_train),
        "eval_unique_db_count": len(required_eval),
        "union_unique_db_count": len(all_required),
        "resolved_db_count": sum(1 for e in databases.values() if e["file_exists"]),
        "missing_required_db_ids": missing,
        "read_only_open_failures": read_only_failures,
        "integrity_failures": integrity_failures,
        "schema_introspection_failures": introspection_failures,
        "schema_hash_missing": schema_missing,
        "databases": databases,
        "hash_conflicts": hash_conflicts,
    }
    return manifest


def _db_schema_set_hash(manifest: dict) -> str:
    entries = []
    for db_id in sorted(manifest["databases"]):
        e = manifest["databases"][db_id]
        entries.append(
            {
                "db_id": db_id,
                "sha256": e.get("sha256"),
                "schema_hash": e.get("schema_hash"),
            }
        )
    canonical = json.dumps(entries, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_text(canonical)


def _content_key(record: dict) -> tuple[str, str]:
    """(db_id, question) key used for historical selection and leakage."""
    return str(record["db_id"]), str(record["question"]).strip()


def _identity_key(record: dict) -> tuple[str, str, str, str]:
    """Full content identity: unique per raw row (db_id, raw question, evidence, SQL)."""
    return (
        str(record["db_id"]),
        str(record["question"]),
        str(record.get("evidence") or "").strip(),
        str(record.get("SQL") or record.get("gold_sql") or "").strip(),
    )


def _old_id_for_record(record: dict, old_by_key: dict[tuple[str, str], str]) -> str | None:
    return old_by_key.get(_content_key(record))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--mini", type=Path, default=DEFAULT_MINI)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--old-engineering",
        type=Path,
        default=DEFAULT_SPLITS / "validation_tuning.jsonl",
    )
    parser.add_argument(
        "--old-formal-validation",
        type=Path,
        default=DEFAULT_SPLITS / "formal_validation.jsonl",
    )
    args = parser.parse_args()

    raw = load_bird_train_filtered(args.raw)
    mini = load_bird_mini_dev(args.mini)
    assert len(raw) == 6601, f"raw train count {len(raw)} != 6601"
    assert len(mini) == 500, f"mini-dev count {len(mini)} != 500"

    raw_sha = sha256_file(args.raw)
    mini_sha = sha256_file(args.mini)
    db_paths = _db_paths_from_registry(args.registry)

    # Existing selections are frozen by content, not by old IDs.  The raw
    # source has one duplicate (db_id, question) pair whose two rows differ by
    # evidence/SQL, so we match selections by full identity, never by key.
    old_eng = _load_jsonl(args.old_engineering)
    old_formal_val = _load_jsonl(args.old_formal_validation)
    eng_identity = {_identity_key(r) for r in old_eng}
    formal_val_identity = {_identity_key(r) for r in old_formal_val}
    assert len(eng_identity) == 120, f"engineering selection has {len(eng_identity)} unique rows"
    assert len(formal_val_identity) == 256, (
        f"formal validation selection has {len(formal_val_identity)} unique rows"
    )
    assert not (eng_identity & formal_val_identity)

    # Build old ID lookup for migration.
    old_eng_by_identity = {_identity_key(r): r["example_id"] for r in old_eng}
    old_fv_by_identity = {_identity_key(r): r["example_id"] for r in old_formal_val}
    old_train_ids_by_identity: dict[tuple[str, str, str, str], str] = {}
    old_train_manifest_path = PROJECT_ROOT / "artifacts/formal_readiness/formal_train_manifest.json"
    if old_train_manifest_path.exists():
        old_train_manifest = json.loads(old_train_manifest_path.read_text(encoding="utf-8"))
        # The manifest only has IDs, not keys.  Reconstruct by matching stable
        # old IDs generated as bird-train-{db}-{sha256(db|question)[:8]}.
        old_by_old_id = {}
        for rec in raw:
            digest = sha256_text(f"{rec['db_id']}|{rec['question']}")[:8]
            old_id = f"bird-train-{rec['db_id']}-{digest}"
            old_by_old_id[old_id] = _identity_key(rec)
        for old_id in old_train_manifest.get("example_ids", []):
            identity = old_by_old_id.get(old_id)
            if identity is not None:
                old_train_ids_by_identity[identity] = old_id

    # Build per-row migration and example rows.
    migration: list[dict] = []
    examples: list[dict] = []
    for idx, rec in enumerate(raw):
        stable_id = canonical_bird_example_id(rec)
        identity = _identity_key(rec)
        old_id = (
            old_eng_by_identity.get(identity)
            or old_fv_by_identity.get(identity)
            or old_train_ids_by_identity.get(identity)
        )
        db_id = rec["db_id"]
        schema_text = schema_for_db(db_paths[db_id])
        if identity in eng_identity:
            split = "validation_tuning"
        elif identity in formal_val_identity:
            split = "formal_validation"
        else:
            split = "train"
        migration.append(
            {
                "source_row_index": idx,
                "old_id_if_any": old_id,
                "stable_id": stable_id,
                "content_signature": bird_content_signature(rec),
                "role": split,
            }
        )
        examples.append(
            {
                "example_id": stable_id,
                "db_id": db_id,
                "question": rec["question"].strip(),
                "schema_text": schema_text,
                "gold_sql": rec.get("SQL") or rec.get("gold_sql") or "",
                "split": split,
                "source": "birdsql/bird23-train-filtered",
                "source_version": "bird23-v1",
                "evidence": rec.get("evidence") or "",
            }
        )

    eng_rows = [e for e in examples if e["split"] == "validation_tuning"]
    fv_rows = [e for e in examples if e["split"] == "formal_validation"]
    train_rows = [e for e in examples if e["split"] == "train"]
    assert len(eng_rows) == 120
    assert len(fv_rows) == 256
    assert len(train_rows) == 6225
    assert len(eng_rows) + len(fv_rows) + len(train_rows) == 6601
    eng_ids = {r["example_id"] for r in eng_rows}
    fv_ids = {r["example_id"] for r in fv_rows}
    train_ids = {r["example_id"] for r in train_rows}
    assert not (eng_ids & fv_ids)
    assert not (eng_ids & train_ids)
    assert not (fv_ids & train_ids)
    assert len(eng_ids | fv_ids | train_ids) == 6601

    # Write canonical split files.  The compatibility alias validation_tuning
    # is kept in sync with engineering_validation.jsonl.
    _write_jsonl(args.splits_dir / "engineering_validation.jsonl", eng_rows)
    _write_jsonl(args.splits_dir / "validation_tuning.jsonl", eng_rows)
    _write_jsonl(args.splits_dir / "formal_validation.jsonl", fv_rows)
    _write_jsonl(args.splits_dir / "formal_train.jsonl", train_rows)

    # Migration manifest.
    _write_jsonl(args.artifact_dir / "formal_id_migration.jsonl", migration)
    assert len({m["stable_id"] for m in migration}) == 6601, "stable IDs not unique"

    # Helper for manifests.
    def _db_dist(rows: list[dict]) -> dict[str, int]:
        return dict(sorted(Counter(r["db_id"] for r in rows).items()))

    # Split manifests.
    eng_manifest = {
        "name": "engineering_validation",
        "count": len(eng_rows),
        "example_ids": [r["example_id"] for r in eng_rows],
        "db_distribution": _db_dist(eng_rows),
        "frozen": True,
    }
    fv_manifest = {
        "name": "formal_validation",
        "seed": 20260818,
        "algorithm_version": "db-stratified-round-robin-v1",
        "count": len(fv_rows),
        "example_ids": [r["example_id"] for r in fv_rows],
        "db_distribution": _db_dist(fv_rows),
        "source_hash": raw_sha,
        "frozen": True,
    }
    train_manifest = {
        "name": "formal_train",
        "count": len(train_rows),
        "example_ids": [r["example_id"] for r in train_rows],
        "db_distribution": _db_dist(train_rows),
        "ids_hash": sha256_text("\n".join(sorted(train_ids))),
        "frozen": True,
    }
    _write_json(args.artifact_dir / "engineering_validation_manifest.json", eng_manifest)
    _write_json(args.artifact_dir / "formal_validation_manifest.json", fv_manifest)
    _write_json(args.artifact_dir / "formal_train_manifest.json", train_manifest)
    _write_json(
        args.splits_dir / "split_manifest.json", _split_manifest(eng_ids, fv_ids, train_ids)
    )

    # Mini-Dev final 500 manifest with canonical official question IDs.
    exposed = _load_exposed()
    exposed_ids = {e["example_id"] for e in exposed}
    mini_ids = [canonical_bird_example_id(m) for m in mini]
    mini_manifest = {
        "count": len(mini),
        "example_ids": mini_ids,
        "db_distribution": _db_dist(mini),
        "difficulty_distribution": dict(
            sorted(Counter(m.get("difficulty", "unknown") for m in mini).items())
        ),
        "exposed_intersection": sorted(exposed_ids & set(mini_ids)),
        "unexposed_count": len(set(mini_ids) - exposed_ids),
    }
    _write_json(args.artifact_dir / "final_select500_manifest.json", mini_manifest)

    # DB registry manifest and schema-set hash.
    db_manifest = _build_db_registry_manifest(args.registry)
    if db_manifest["hash_conflicts"]:
        raise SystemExit(f"unexpected DB hash conflicts: {db_manifest['hash_conflicts']}")
    if db_manifest["missing_required_db_ids"]:
        raise SystemExit(f"missing DBs: {db_manifest['missing_required_db_ids']}")
    if db_manifest["integrity_failures"]:
        raise SystemExit(f"quick_check failures: {db_manifest['integrity_failures']}")
    if db_manifest["schema_hash_missing"]:
        raise SystemExit(f"missing schema hashes: {db_manifest['schema_hash_missing']}")
    db_schema_set_hash = _db_schema_set_hash(db_manifest)
    _write_json(args.artifact_dir / "db_registry_manifest.json", db_manifest)

    # Leakage audit using stable IDs and content hashes.
    mini_set = set(mini_ids)
    leakage = _leakage_report(raw, mini, eng_rows, fv_rows, train_rows, mini_set)
    _write_json(args.artifact_dir / "leakage_report.json", leakage)

    # Formal protocol lock.
    lock = _protocol_lock(
        raw_sha=raw_sha,
        mini_sha=mini_sha,
        artifact_dir=args.artifact_dir,
        db_manifest_sha=sha256_file(args.artifact_dir / "db_registry_manifest.json"),
        db_schema_set_sha=db_schema_set_hash,
        eng_manifest=eng_manifest,
        fv_manifest=fv_manifest,
        train_manifest=train_manifest,
        mini_manifest=mini_manifest,
    )
    _write_json(args.artifact_dir / "formal_protocol_lock.json", lock)

    print("formal_train.jsonl", len(train_rows))
    print("engineering_validation.jsonl", len(eng_rows))
    print("formal_validation.jsonl", len(fv_rows))
    print("stable_id_unique", len({m["stable_id"] for m in migration}))
    print(
        "db_registry_resolved",
        db_manifest["resolved_db_count"],
        "/",
        len(db_manifest["all_required_db_ids"]),
    )
    print("db_schema_set_sha256", db_schema_set_hash)
    print("leakage", leakage)


def _split_manifest(eng_ids: set[str], fv_ids: set[str], train_ids: set[str]) -> dict:
    return {
        "full_filtered_train_count": len(eng_ids | fv_ids | train_ids),
        "engineering_validation_count": len(eng_ids),
        "formal_validation_count": len(fv_ids),
        "formal_train_count": len(train_ids),
        "sum_check": len(eng_ids) + len(fv_ids) + len(train_ids),
        "union_equals_6601": len(eng_ids | fv_ids | train_ids) == 6601,
        "overlaps": {
            "eng_vs_formal_val": len(eng_ids & fv_ids),
            "eng_vs_formal_train": len(eng_ids & train_ids),
            "formal_val_vs_formal_train": len(fv_ids & train_ids),
        },
    }


def _load_exposed() -> list[dict]:
    path = PROJECT_ROOT / "artifacts/experiment/benchmark_exposure_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("examples", [])


def _leakage_report(
    raw: list[dict],
    mini: list[dict],
    eng_rows: list[dict],
    fv_rows: list[dict],
    train_rows: list[dict],
    mini_ids: set[str],
) -> dict:
    from querydistill.data.leakage import normalize_question

    groups = {
        "formal_train": train_rows,
        "engineering_validation": eng_rows,
        "formal_validation": fv_rows,
    }
    # Content-level comparisons use raw records to avoid depending on IDs.
    train_keys = {(r["db_id"], r["question"].strip()) for r in train_rows}
    eng_keys = {(r["db_id"], r["question"].strip()) for r in eng_rows}
    fv_keys = {(r["db_id"], r["question"].strip()) for r in fv_rows}

    def _question_evidence_hash(r: dict) -> str:
        return sha256_text(f"{r['db_id']}|{r['question']}|{r.get('evidence', '')}")

    def _question_sql_hash(r: dict) -> str:
        return sha256_text(f"{r['db_id']}|{r['question']}|{r.get('SQL', '')}")

    id_overlap = 0
    question_overlap = 0
    qe_overlap = 0
    qsql_overlap = 0
    for _left_name, left_rows in groups.items():
        left_ids = {r["example_id"] for r in left_rows}
        if left_ids & mini_ids:
            id_overlap += len(left_ids & mini_ids)
        left_q = {normalize_question(r["question"]) for r in left_rows}
        right_q = {normalize_question(m["question"]) for m in mini}
        if left_q & right_q:
            question_overlap += len(left_q & right_q)
        left_qe = {_question_evidence_hash(r) for r in left_rows}
        right_qe = {_question_evidence_hash(m) for m in mini}
        if left_qe & right_qe:
            qe_overlap += len(left_qe & right_qe)
        left_qsql = {_question_sql_hash(r) for r in left_rows}
        right_qsql = {_question_sql_hash(m) for m in mini}
        if left_qsql & right_qsql:
            qsql_overlap += len(left_qsql & right_qsql)

    # Pairwise split overlap on stable IDs and content keys.
    pairwise_id_overlap = 0
    pairwise_key_overlap = 0
    split_sets = [("train", train_keys), ("eng", eng_keys), ("fv", fv_keys)]
    for i in range(len(split_sets)):
        for j in range(i + 1, len(split_sets)):
            _, a = split_sets[i]
            _, b = split_sets[j]
            pairwise_key_overlap += len(a & b)

    dup_exact = len(raw) - len({(r["db_id"], r["question"].strip()) for r in raw})
    dup_normalized = len(raw) - len({(r["db_id"], normalize_question(r["question"])) for r in raw})

    return {
        "id_overlap_with_mini_dev": id_overlap,
        "normalized_question_overlap_with_mini_dev": question_overlap,
        "question_evidence_hash_overlap_with_mini_dev": qe_overlap,
        "question_gold_sql_hash_overlap_with_mini_dev": qsql_overlap,
        "pairwise_split_id_overlap": pairwise_id_overlap,
        "pairwise_split_content_overlap": pairwise_key_overlap,
        "duplicate_exact_question_count_within_train_source": dup_exact,
        "duplicate_normalized_question_count_within_train_source": dup_normalized,
        "mini_dev_exposed_count": len(_load_exposed()),
        "mini_dev_total": len(mini),
    }


def _protocol_lock(
    raw_sha: str,
    mini_sha: str,
    artifact_dir: Path,
    db_manifest_sha: str,
    db_schema_set_sha: str,
    eng_manifest: dict,
    fv_manifest: dict,
    train_manifest: dict,
    mini_manifest: dict,
) -> dict:
    def _manifest_sha(name: str, payload: dict) -> str:
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return sha256_text(canonical)

    from querydistill.artifacts.manifest import config_hash

    # The executor config hash is computed from the current SafeSQLExecutor
    # defaults; keep it stable and explicit.
    from querydistill.sql.executor import SafeSQLExecutor

    executor_cfg = {
        "max_rows": 1000,
        "max_execution_ms": 3000,
        "worker_start_timeout": 30.0,
        "read_only_uri": True,
        "authorizer": "select/read/function only",
    }
    verifier_cfg = {"mode": "multiset default, ordered when ORDER BY semantic"}
    reward_cfg = {"components": ["format", "parse", "safety", "execution", "correctness"]}

    return {
        "format_version": "1.9",
        "git_commit": "PENDING_FINAL_COMMIT",
        "dataset": {
            "filtered_train_sha256": raw_sha,
            "mini_dev_sha256": mini_sha,
        },
        "splits": {
            "formal_train_manifest_sha256": _manifest_sha("formal_train_manifest", train_manifest),
            "engineering_validation_manifest_sha256": _manifest_sha(
                "engineering_validation_manifest", eng_manifest
            ),
            "formal_validation_manifest_sha256": _manifest_sha(
                "formal_validation_manifest", fv_manifest
            ),
            "final_select500_manifest_sha256": _manifest_sha(
                "final_select500_manifest", mini_manifest
            ),
        },
        "database": {
            "db_registry_hash": db_manifest_sha,
            "db_registry_manifest_sha256": db_manifest_sha,
            "db_schema_set_sha256": db_schema_set_sha,
        },
        "models": {
            "teacher_model_id": "Qwen/Qwen3-4B",
            "teacher_revision": "resolved",
            "student_model_id": "Qwen/Qwen3-0.6B-Base",
            "student_revision": "resolved",
        },
        "prompts": {
            "teacher_prompt_version": "bird-v2",
            "student_prompt_version": "bird-v1",
        },
        "protocol": {
            "qwen_chat_template": True,
            "sql_only": True,
            "sql_stopping_policy": "StopAfterSqlClose + SqlStoppingGRPOTrainer",
            "sql_stopping_version": "canonical-</sql>",
        },
        "execution": {
            "gold_audit_timeout_ms": 30000,
            "candidate_execution_timeout_ms": SafeSQLExecutor.__init__.__defaults__[1],
            "candidate_timeout_frozen": False,
        },
        "safety": {
            "SafeSQLExecutor_config_hash": config_hash(executor_cfg),
            "scalar_replace_allowed": True,
            "replace_into_forbidden": True,
            "layer2_read_only_authorizer": True,
        },
        "verification": {
            "ResultEquivalenceVerifier_config_hash": config_hash(verifier_cfg),
        },
        "reward": {
            "CompositeReward_config_hash": config_hash(reward_cfg),
        },
        "training_exclusions": {
            "engineering_validation_forbidden": True,
            "formal_validation_forbidden": True,
            "mini_dev_forbidden": True,
        },
        "created_at": __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(),
    }


if __name__ == "__main__":
    main()
