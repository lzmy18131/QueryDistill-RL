#!/usr/bin/env python3
"""Full 6601 Gold SQL execution audit (read-only, no result equivalence)."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from querydistill.sql.environment import SQLExecutionEnvironment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output-audit", required=True)
    parser.add_argument("--output-failures", required=True)
    parser.add_argument("--max-examples", type=int, default=6601)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.examples).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.max_examples]
    env = SQLExecutionEnvironment.from_registry(args.registry)
    audit = {
        "total": len(records),
        "execution_success": 0,
        "execution_error": 0,
        "timeout": 0,
        "unsafe": 0,
        "latency_sum_ms": 0.0,
        "failures": [],
    }
    failure_path = Path(args.output_failures)
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with failure_path.open("w", encoding="utf-8") as f:
        for idx, r in enumerate(records, 1):
            db_id = r["db_id"]
            sql = r.get("SQL") or r.get("gold_sql") or ""
            example_id = (
                r.get("example_id")
                or f"bird-train-{db_id}-{hashlib.sha256((db_id + r['question']).encode()).hexdigest()[:8]}"
            )
            t0 = time.time()
            try:
                result = env.execute(db_id, sql)
            except Exception as exc:  # noqa: BLE001
                result = None
                error_type = f"internal:{type(exc).__name__}"
                audit["execution_error"] += 1
                rec = {
                    "example_id": example_id,
                    "db_id": db_id,
                    "error_type": error_type,
                    "sanitized_error": str(exc)[:500],
                }
                audit["failures"].append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            duration_ms = (time.time() - t0) * 1000
            audit["latency_sum_ms"] += duration_ms
            if result.success:
                audit["execution_success"] += 1
            else:
                audit["execution_error"] += 1
                if result.timed_out:
                    audit["timeout"] += 1
                if result.error_type in {"unsafe", "safety"}:
                    audit["unsafe"] += 1
                rec = {
                    "example_id": example_id,
                    "db_id": db_id,
                    "error_type": result.error_type,
                    "sanitized_error": result.error_message[:500],
                }
                audit["failures"].append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if idx % 500 == 0:
                print(f"progress {idx}/{len(records)}", flush=True)
    audit["duration_seconds"] = round(time.time() - start, 2)
    audit["mean_latency_ms"] = round(audit["latency_sum_ms"] / max(1, len(records)), 2)
    Path(args.output_audit).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
