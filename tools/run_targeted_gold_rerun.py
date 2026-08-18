#!/usr/bin/env python3
"""Targeted Gold reruns for the Phase 1.9 gate.

Executes only a named historical failure bucket (airline or replace_safety)
against the current DB registry and safety policy, using the explicit Gold
audit timeout (default 30000 ms).

Unlike an inline Python snippet, this file-based entry point is compatible with
the multiprocessing spawn context used by SafeSQLExecutor on Windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from querydistill.data.bird import canonical_bird_example_id, load_bird_train_filtered
from querydistill.sql.environment import SQLExecutionEnvironment


def _old_audit_id(record: dict) -> str:
    digest = hashlib.sha256((record["db_id"] + record["question"]).encode("utf-8")).hexdigest()[:8]
    return f"bird-train-{record['db_id']}-{digest}"


def _load_historical_failures(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", choices=["airline", "replace_safety"], required=True)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument(
        "--failures", required=True, help="Historical gold_execution_failures.jsonl"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    args = parser.parse_args()

    raw = load_bird_train_filtered(args.examples)
    old_by_id = {_old_audit_id(r): r for r in raw}
    failures = _load_historical_failures(Path(args.failures))

    if args.bucket == "airline":
        records = [r for r in raw if r["db_id"] == "airline"]
    else:
        unsafe_ids = [f["example_id"] for f in failures if f["error_type"] == "unsafe_sql"]
        records = []
        for old_id in unsafe_ids:
            rec = old_by_id.get(old_id)
            if rec is None:
                raise SystemExit(f"missing historical unsafe record {old_id}")
            records.append(rec)

    env = SQLExecutionEnvironment.from_registry(
        args.registry, max_rows=1000, max_execution_ms=args.timeout_ms
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for rec in records:
        stable_id = canonical_bird_example_id(rec)
        sql = rec.get("SQL") or rec.get("gold_sql") or ""
        t0 = time.time()
        try:
            result = env.execute(rec["db_id"], sql)
        except Exception as exc:  # noqa: BLE001 - classify, do not crash
            result = None
            exc_text = f"internal:{type(exc).__name__}: {exc}"
        else:
            exc_text = ""
        duration_ms = round((time.time() - t0) * 1000.0, 2)
        rows.append(
            {
                "example_id": stable_id,
                "old_example_id": _old_audit_id(rec),
                "db_id": rec["db_id"],
                "success": bool(result and result.success),
                "error_type": result.error_type if result else exc_text,
                "error_message": (
                    result.error_message[:500] if result and not result.success else ""
                ),
                "duration_ms": duration_ms,
                "timed_out": bool(result and result.timed_out),
            }
        )
    out_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "bucket": args.bucket,
                "total": len(rows),
                "success": sum(1 for r in rows if r["success"]),
                "error_types": {
                    et: sum(1 for r in rows if r["error_type"] == et)
                    for et in set(r["error_type"] for r in rows)
                },
                "output": str(out_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
