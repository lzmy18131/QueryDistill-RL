#!/usr/bin/env python3
"""Build final Gold execution artifacts from the full 30s audit.

Reads ``gold_execution_results.jsonl`` and the historical
``gold_execution_failures.jsonl`` to produce:

* ``gold_execution_final.json`` - full classification table
* ``gold_latency_profile.json`` - latency percentiles and buckets
* ``execution_policy_compatibility.json`` - candidate timeout compatibility
* ``timeout_10s_rerun.jsonl`` / ``timeout_30s_rerun.jsonl`` - staged buckets
  derived from the richer single 30s run
* ``gold_slow_query_candidates.jsonl`` - residual 30s timeouts for 120s
  diagnostics
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from querydistill.data.bird import canonical_bird_example_id

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _old_audit_id(record: dict) -> str:
    digest = hashlib.sha256((record["db_id"] + record["question"]).encode("utf-8")).hexdigest()[:8]
    return f"bird-train-{record['db_id']}-{digest}"


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = f + 1
    if c >= len(sorted_values):
        return sorted_values[-1]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Full audit results JSONL.")
    parser.add_argument(
        "--historical-audit", required=True, help="Historical gold_execution_audit.json."
    )
    parser.add_argument(
        "--historical-failures", required=True, help="Historical gold_execution_failures.jsonl."
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts/final_pretraining_gate"
    )
    parser.add_argument("--airline-rerun", type=Path)
    parser.add_argument("--replace-rerun", type=Path)
    parser.add_argument("--slow-diagnostic", type=Path)
    parser.add_argument("--migration", type=Path)
    parser.add_argument(
        "--examples",
        type=Path,
        default=PROJECT_ROOT / "data/bird/raw/bird23_train_filtered.jsonl",
    )
    args = parser.parse_args()

    results = _load_jsonl(Path(args.results))
    historical_audit = json.loads(Path(args.historical_audit).read_text(encoding="utf-8"))
    historical_failures = _load_jsonl(Path(args.historical_failures))
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Full audit outcome categories.
    success = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    timeouts_30s = [r for r in failures if r["error_type"] == "timeout"]
    slow_diag = _load_jsonl(args.slow_diagnostic) if args.slow_diagnostic else []
    slow_success_count = sum(1 for r in slow_diag if r.get("success"))
    persistent_timeout_count = sum(
        1 for r in slow_diag if not r.get("success") and r.get("timed_out", False)
    )
    sqlite_errors = [r for r in failures if r["error_type"] == "sqlite_error"]
    unsafe = [r for r in failures if r["error_type"] in {"unsafe", "unsafe_sql"}]
    internal = [r for r in failures if r["error_type"] == "internal_error"]
    other = [
        r
        for r in failures
        if r["error_type"]
        not in {"timeout", "sqlite_error", "unsafe", "unsafe_sql", "internal_error"}
    ]

    migration = _load_jsonl(args.migration) if args.migration else []
    stable_by_old = {
        m["old_id_if_any"]: m["stable_id"] for m in migration if m.get("old_id_if_any")
    }
    raw_records = _load_jsonl(args.examples)
    old_audit_to_stable = {_old_audit_id(r): canonical_bird_example_id(r) for r in raw_records}
    historical_timeout_ids = {
        stable_by_old.get(
            f["example_id"], old_audit_to_stable.get(f["example_id"], f["example_id"])
        )
        for f in historical_failures
        if f["error_type"] == "timeout"
    }
    historical_timeout_results = [r for r in results if r["example_id"] in historical_timeout_ids]
    timeout_10s = []
    timeout_30s = []
    for r in historical_timeout_results:
        r10 = dict(r)
        r10["stage_a_10s_outcome"] = (
            "success" if r["success"] and r["duration_ms"] <= 10000 else "timeout"
        )
        timeout_10s.append(r10)
        r30 = dict(r)
        if r["success"] and r["duration_ms"] <= 10000:
            r30["stage_b_30s_outcome"] = "success_under_10s"
        elif r["success"]:
            r30["stage_b_30s_outcome"] = "success_10_to_30s"
        else:
            r30["stage_b_30s_outcome"] = "timeout_30s"
        timeout_30s.append(r30)

    # Airline / replace buckets from the full audit (fall back to targeted rerun
    # if supplied for provenance).
    airline_ids = {r["example_id"] for r in _load_jsonl(args.airline_rerun) if args.airline_rerun}
    replace_ids = {r["example_id"] for r in _load_jsonl(args.replace_rerun) if args.replace_rerun}
    if airline_ids:
        airline_results = [r for r in results if r["example_id"] in airline_ids]
    else:
        airline_results = [r for r in results if r["db_id"] == "airline"]
    if replace_ids:
        replace_results = [r for r in results if r["example_id"] in replace_ids]
    else:
        unsafe_old_ids = {
            f["example_id"] for f in historical_failures if f["error_type"] == "unsafe_sql"
        }
        replace_results = [r for r in results if r["example_id"] in unsafe_old_ids]

    # Latency profile from every successful full-audit execution.
    durations = [r["duration_ms"] for r in success if r.get("duration_ms") is not None]
    latency_profile = {
        "sample_count": len(durations),
        "P50_ms": round(_percentile(durations, 0.50), 2),
        "P90_ms": round(_percentile(durations, 0.90), 2),
        "P95_ms": round(_percentile(durations, 0.95), 2),
        "P99_ms": round(_percentile(durations, 0.99), 2),
        "max_ms": round(max(durations), 2) if durations else 0.0,
        "count_le_3s": sum(1 for d in durations if d <= 3000),
        "count_gt_3s": sum(1 for d in durations if d > 3000),
        "count_le_5s": sum(1 for d in durations if d <= 5000),
        "count_gt_5s": sum(1 for d in durations if d > 5000),
        "count_le_10s": sum(1 for d in durations if d <= 10000),
        "count_gt_10s": sum(1 for d in durations if d > 10000),
        "count_le_30s": sum(1 for d in durations if d <= 30000),
        "count_gt_30s": sum(1 for d in durations if d > 30000),
        "slowest_dbs": _slowest_dbs(results),
    }

    # Candidate timeout compatibility (Gold reference queries that would exceed
    # a candidate timeout if it were used as the Gold audit timeout).
    compatibility = {}
    for threshold_ms, label in [(3000, "3s"), (5000, "5s"), (10000, "10s")]:
        exceeded = [
            r
            for r in results
            if (r.get("duration_ms") or 0) > threshold_ms or r.get("timed_out", False)
        ]
        compatibility[label] = {
            "gold_queries_exceeding": len(exceeded),
            "total_gold_queries": len(results),
            "ratio": round(len(exceeded) / max(1, len(results)), 4),
        }

    final = {
        "source_rows": len(results),
        "historical_success_under_3s": historical_audit.get("execution_success", 0),
        "historical_timeout_count": historical_audit.get("timeout", 0),
        "fixed_airline_success": sum(1 for r in airline_results if r["success"]),
        "fixed_airline_total": len(airline_results),
        "fixed_replace_success": sum(1 for r in replace_results if r["success"]),
        "fixed_replace_total": len(replace_results),
        "success_under_3s_actual": sum(1 for r in success if r["duration_ms"] <= 3000),
        "success_3_to_10s": sum(1 for r in success if 3000 < r["duration_ms"] <= 10000),
        "success_10_to_30s": sum(1 for r in success if 10000 < r["duration_ms"] <= 30000),
        "slow_success_30_to_120s": slow_success_count,
        "normal_success_under_30s": len(success),
        "total_success_including_slow": len(success) + slow_success_count,
        "persistent_timeout_30s": len(timeouts_30s),
        "persistent_timeout_after_120s": persistent_timeout_count,
        "sqlite_error": len(sqlite_errors),
        "unsafe": len(unsafe),
        "internal_error": len(internal),
        "other": len(other),
        "executable_or_classified": len(results),
        "audit_timeout_ms": 30000,
    }

    _write_json(artifact_dir / "gold_execution_final.json", final)
    _write_json(artifact_dir / "gold_latency_profile.json", latency_profile)
    _write_json(artifact_dir / "execution_policy_compatibility.json", compatibility)
    _write_jsonl(artifact_dir / "timeout_10s_rerun.jsonl", timeout_10s)
    _write_jsonl(artifact_dir / "timeout_30s_rerun.jsonl", timeout_30s)
    _write_jsonl(
        artifact_dir / "gold_slow_query_candidates.jsonl",
        [
            {
                "example_id": r["example_id"],
                "db_id": r["db_id"],
                "sql_hash": None,
                "timeout_ms": 30000,
            }
            for r in timeouts_30s
        ],
    )

    print(json.dumps(final, ensure_ascii=False, indent=2))
    print("latency", json.dumps(latency_profile, ensure_ascii=False))
    print("compatibility", json.dumps(compatibility, ensure_ascii=False))


def _slowest_dbs(results: list[dict]) -> list[dict]:
    by_db: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if r.get("duration_ms") is not None:
            by_db[r["db_id"]].append(r["duration_ms"])
    rows = []
    for db_id, vals in by_db.items():
        rows.append(
            {
                "db_id": db_id,
                "count": len(vals),
                "mean_ms": round(sum(vals) / len(vals), 2),
                "max_ms": round(max(vals), 2),
            }
        )
    rows.sort(key=lambda x: x["mean_ms"], reverse=True)
    return rows[:20]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
