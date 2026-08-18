#!/usr/bin/env python3
"""Optional 120s diagnostic for Gold queries that remain timeout at 30s.

Only run when the residual 30s timeout count is small (<=100).  The purpose is
to distinguish slow-but-executable Gold SQL from never-finishing/pathological
queries.  It uses the same standalone subprocess worker as the full audit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from querydistill.data.bird import canonical_bird_example_id, load_bird_train_filtered

_WORKER = Path(__file__).resolve().parent / "gold_execution_worker.py"
_TMP_ROOT = Path("D:/LLMCache/phase1_9/tmp")


def _load_registry(registry_path: Path) -> dict[str, Path]:
    payload = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    raw = payload.get("databases", payload)
    paths: dict[str, Path] = {}
    for db_id, rel in raw.items():
        p = Path(str(rel))
        if not p.is_absolute():
            p = Path(registry_path).parent / p
        paths[db_id] = p.resolve()
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="gold_slow_query_candidates.jsonl")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout-ms", type=int, default=120000)
    args = parser.parse_args()

    candidates = [
        json.loads(line)
        for line in Path(args.candidates).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(candidates) > 100:
        raise SystemExit(
            f"residual timeout count {len(candidates)} > 100; do not auto-run 120s diagnostic"
        )

    raw = load_bird_train_filtered(args.examples)
    by_stable = {canonical_bird_example_id(r): r for r in raw}
    db_paths = _load_registry(Path(args.registry))
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []
    for cand in candidates:
        stable_id = cand["example_id"]
        rec = by_stable.get(stable_id)
        if rec is None:
            rows.append(
                {
                    "example_id": stable_id,
                    "success": False,
                    "error_type": "internal_error",
                    "error_message": "candidate not found in raw source",
                    "duration_ms": 0.0,
                    "timed_out": False,
                }
            )
            continue
        sql = rec.get("SQL") or rec.get("gold_sql") or ""
        db_path = db_paths.get(rec["db_id"])
        t0 = time.time()
        if db_path is None or not db_path.is_file():
            rows.append(
                {
                    "example_id": stable_id,
                    "db_id": rec["db_id"],
                    "success": False,
                    "error_type": "schema_error",
                    "error_message": "db missing",
                    "duration_ms": round((time.time() - t0) * 1000, 2),
                    "timed_out": False,
                }
            )
            continue
        sql_file = _TMP_ROOT / f"slow_sql_{time.monotonic_ns()}.sql"
        out_file = _TMP_ROOT / f"slow_out_{time.monotonic_ns()}.json"
        sql_file.write_text(sql, encoding="utf-8")
        try:
            subprocess.run(
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
                    str(args.timeout_ms),
                    "--output-json",
                    str(out_file),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=(args.timeout_ms / 1000.0) + 30.0,
                check=False,
            )
            if out_file.exists():
                result = json.loads(out_file.read_text(encoding="utf-8"))
            else:
                result = {
                    "success": False,
                    "error_type": "internal_error",
                    "error_message": "worker produced no output",
                    "timed_out": False,
                }
        except subprocess.TimeoutExpired:
            result = {
                "success": False,
                "error_type": "timeout",
                "error_message": "worker subprocess exceeded 120s diagnostic deadline",
                "timed_out": True,
            }
        finally:
            sql_file.unlink(missing_ok=True)
            out_file.unlink(missing_ok=True)

        rows.append(
            {
                "example_id": stable_id,
                "db_id": rec["db_id"],
                "success": result.get("success", False),
                "error_type": result.get("error_type", "internal_error"),
                "error_message": str(result.get("error_message", ""))[:500],
                "duration_ms": round((time.time() - t0) * 1000, 2),
                "timed_out": bool(result.get("timed_out", False)),
                "diagnostic_timeout_ms": args.timeout_ms,
            }
        )

    Path(args.output).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "total": len(rows),
                "success": sum(1 for r in rows if r["success"]),
                "persistent_timeout": sum(1 for r in rows if not r["success"] and r["timed_out"]),
                "sqlite_error": sum(
                    1 for r in rows if not r["success"] and r["error_type"] == "sqlite_error"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
