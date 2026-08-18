#!/usr/bin/env python3
"""Final Gold SQL execution audit for the Phase 1.9 pretraining gate.

This tool executes Gold SQL with an explicit correctness-audit timeout
(``--audit-timeout-ms``, default 30000 ms).  It records every example's result
and latency so downstream artifacts can build the full Gold latency profile.

The candidate / GRPO execution timeout is intentionally kept separate; this
tool never reads or modifies that policy.

Each query is executed by a standalone subprocess worker
(``tools/gold_execution_worker.py``) so the audit works even in Windows
sandboxes where ``multiprocessing`` spawn handle duplication is denied.  The
worker enforces the same read-only SQLite authorizer and timeout semantics as
SafeSQLExecutor's worker; the timeout is passed explicitly from this tool.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from querydistill.data.bird import canonical_bird_example_id
from querydistill.sql.safety import validate_sql

# The correctness audit timeout is a distinct policy from the candidate/RL
# execution timeout.  This constant is only the CLI default for this tool.
DEFAULT_AUDIT_TIMEOUT_MS = 30000

_WORKER = Path(__file__).resolve().parent / "gold_execution_worker.py"
_TMP_ROOT = Path(os.environ.get("GOLD_AUDIT_TMP", "D:/LLMCache/phase1_9/tmp"))


def _load_registry_db_paths(registry_path: Path) -> dict[str, Path]:
    payload = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    raw = payload.get("databases", payload)
    paths: dict[str, Path] = {}
    for db_id, rel in raw.items():
        path = Path(str(rel))
        if not path.is_absolute():
            path = Path(registry_path).parent / path
        paths[db_id] = path.resolve()
    return paths


def _run_worker(db_path: Path, sql: str, timeout_ms: int, tmp_dir: Path) -> dict:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    sql_file = tmp_dir / f"sql_{os.getpid()}_{time.monotonic_ns()}.sql"
    out_file = tmp_dir / f"out_{os.getpid()}_{time.monotonic_ns()}.json"
    sql_file.write_text(sql, encoding="utf-8")
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(_WORKER),
                "--db-path",
                str(db_path),
                "--sql-file",
                str(sql_file),
                "--max-rows",
                "1000",
                "--timeout-ms",
                str(timeout_ms),
                "--output-json",
                str(out_file),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=(timeout_ms / 1000.0) + 30.0,
            check=False,
        )
        if proc.returncode != 0 or not out_file.exists():
            return {
                "success": False,
                "error_type": "internal_error",
                "error_message": f"worker exited with code {proc.returncode}",
                "timed_out": False,
            }
        return json.loads(out_file.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error_type": "timeout",
            "error_message": "worker subprocess exceeded hard deadline",
            "timed_out": True,
        }
    finally:
        with contextlib.suppress(FileNotFoundError):
            sql_file.unlink(missing_ok=True)
            out_file.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", required=True, help="Raw BIRD train JSONL (6601 rows).")
    parser.add_argument("--registry", required=True, help="BIRD SQLite db registry JSON.")
    parser.add_argument("--output-audit", required=True, help="Summary JSON output.")
    parser.add_argument("--output-results", required=True, help="Per-example results JSONL.")
    parser.add_argument("--output-failures", required=True, help="Failures-only JSONL.")
    parser.add_argument(
        "--audit-timeout-ms",
        type=int,
        default=DEFAULT_AUDIT_TIMEOUT_MS,
        help="Gold correctness audit timeout in milliseconds (default 30000).",
    )
    parser.add_argument("--max-examples", type=int, default=6601)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Sequential worker count.  This tool currently uses 1; accepted for future use.",
    )
    args = parser.parse_args()

    if args.audit_timeout_ms <= 0:
        parser.error("--audit-timeout-ms must be positive")
    if args.workers not in (1, 2):
        parser.error("--workers must be 1 or 2")

    records = [
        json.loads(line)
        for line in Path(args.examples).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.max_examples]
    db_paths = _load_registry_db_paths(Path(args.registry))

    results_path = Path(args.output_results)
    failures_path = Path(args.output_failures)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)

    start = time.time()
    summary = {
        "total": len(records),
        "execution_success": 0,
        "execution_error": 0,
        "timeout": 0,
        "unsafe": 0,
        "sqlite_error": 0,
        "internal_error": 0,
        "schema_error": 0,
        "other_error": 0,
        "latency_sum_ms": 0.0,
        "audit_timeout_ms": args.audit_timeout_ms,
        "workers": args.workers,
    }

    with (
        results_path.open("w", encoding="utf-8") as res_f,
        failures_path.open("w", encoding="utf-8") as fail_f,
    ):
        for idx, record in enumerate(records, 1):
            db_id = record["db_id"]
            sql = record.get("SQL") or record.get("gold_sql") or ""
            stable_id = canonical_bird_example_id(record)
            t0 = time.time()

            decision = validate_sql(sql)
            if not decision.safe:
                duration_ms = round((time.time() - t0) * 1000.0, 2)
                rec = {
                    "source_row_index": idx - 1,
                    "example_id": stable_id,
                    "db_id": db_id,
                    "success": False,
                    "error_type": decision.error_type,
                    "error_message": decision.reason[:500],
                    "duration_ms": duration_ms,
                    "timed_out": False,
                }
                summary["execution_error"] += 1
                summary["unsafe"] += 1
                summary["latency_sum_ms"] += duration_ms
                res_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fail_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            db_path = db_paths.get(db_id)
            if db_path is None or not db_path.is_file():
                duration_ms = round((time.time() - t0) * 1000.0, 2)
                rec = {
                    "source_row_index": idx - 1,
                    "example_id": stable_id,
                    "db_id": db_id,
                    "success": False,
                    "error_type": "schema_error",
                    "error_message": f"database not found in registry: {db_id}",
                    "duration_ms": duration_ms,
                    "timed_out": False,
                }
                summary["execution_error"] += 1
                summary["schema_error"] += 1
                summary["latency_sum_ms"] += duration_ms
                res_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fail_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            try:
                worker_result = _run_worker(db_path, sql, args.audit_timeout_ms, _TMP_ROOT)
            except Exception as exc:  # noqa: BLE001 - tool must classify all rows
                worker_result = {
                    "success": False,
                    "error_type": "internal_error",
                    "error_message": f"{type(exc).__name__}: {exc}",
                    "timed_out": False,
                }

            duration_ms = (time.time() - t0) * 1000
            summary["latency_sum_ms"] += duration_ms
            rec = {
                "source_row_index": idx - 1,
                "example_id": stable_id,
                "db_id": db_id,
                "success": worker_result.get("success", False),
                "error_type": worker_result.get("error_type", "internal_error"),
                "error_message": str(worker_result.get("error_message", ""))[:500],
                "duration_ms": round(duration_ms, 2),
                "timed_out": bool(worker_result.get("timed_out", False)),
            }
            if rec["success"]:
                summary["execution_success"] += 1
            else:
                summary["execution_error"] += 1
                et = rec["error_type"]
                if et == "timeout":
                    summary["timeout"] += 1
                elif et in {"unsafe", "unsafe_sql"}:
                    summary["unsafe"] += 1
                elif et == "sqlite_error":
                    summary["sqlite_error"] += 1
                elif et == "schema_error":
                    summary["schema_error"] += 1
                elif et == "internal_error":
                    summary["internal_error"] += 1
                else:
                    summary["other_error"] += 1
            res_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if not rec["success"]:
                fail_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if idx % 500 == 0:
                print(f"progress {idx}/{len(records)}", flush=True)

    summary["duration_seconds"] = round(time.time() - start, 2)
    summary["mean_latency_ms"] = round(summary["latency_sum_ms"] / max(1, len(records)), 2)
    summary["sqlite_version"] = _sqlite_version()
    summary["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    Path(args.output_audit).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _sqlite_version() -> str:
    import sqlite3

    return sqlite3.sqlite_version


if __name__ == "__main__":
    main()
