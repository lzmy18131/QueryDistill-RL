#!/usr/bin/env python3
"""Compute Phase 1.5 teacher diagnostic metrics from attempts.jsonl."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from querydistill.utils import atomic_write_json, load_json, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--error-output", required=True)
    args = parser.parse_args()

    ids = load_json(args.ids)
    records = load_jsonl(args.attempts)
    requested_ids = [e["example_id"] for e in ids["examples"]]
    requested = set(requested_ids)

    attempt1 = [r for r in records if r["attempt_index"] == 0]
    retries = [r for r in records if r["attempt_index"] > 0]

    def verified_ids(items):
        return {r["example_id"] for r in items if r["execution_equivalent"]}

    attempt1_verified = verified_ids(attempt1)
    final_verified = verified_ids(records)
    retry_incremental = final_verified - attempt1_verified

    # Duplicate retry: retry candidate SQL equals that retry's own attempt1 SQL.
    attempt1_by_id = {r["example_id"]: r for r in attempt1}
    duplicate_retries = 0
    for r in retries:
        prev = attempt1_by_id.get(r["example_id"])
        if prev is None:
            continue
        if prev.get("candidate_sql") and prev["candidate_sql"] == r.get("candidate_sql"):
            duplicate_retries += 1

    unique_sql = {r.get("normalized_sql_hash") for r in records if r.get("normalized_sql_hash")}

    def rates(items, denom):
        if not denom:
            return 0.0
        return round(len(items) / denom, 4)

    error_counts = Counter()
    for r in records:
        if not r.get("parse_valid") or not r.get("candidate_sql"):
            error_counts["parse_failed"] += 1
        elif not r.get("safe"):
            error_counts["unsafe_or_multiple_statement"] += 1
        elif not r.get("execution_success"):
            error_counts["sqlite_execution_error"] += 1
        elif not r.get("execution_equivalent"):
            error_counts["wrong_result"] += 1
        else:
            error_counts["verified"] += 1

    metrics = {
        "requested_examples": len(requested),
        "attempt1_generated": len(attempt1),
        "retry_generated": len(retries),
        "attempt1_parse_valid_rate": rates(
            [r for r in attempt1 if r.get("parse_valid")], len(attempt1)
        ),
        "attempt1_safe_rate": rates([r for r in attempt1 if r.get("safe")], len(attempt1)),
        "attempt1_execution_success_rate": rates(
            [r for r in attempt1 if r.get("execution_success")], len(attempt1)
        ),
        "attempt1_verified_example_count": len(attempt1_verified),
        "attempt1_verified_example_coverage": round(len(attempt1_verified) / len(requested), 4)
        if requested
        else 0.0,
        "retry_count": len(retries),
        "retry_parse_valid_rate": rates([r for r in retries if r.get("parse_valid")], len(retries)),
        "retry_execution_success_rate": rates(
            [r for r in retries if r.get("execution_success")], len(retries)
        ),
        "retry_incremental_verified_examples": len(retry_incremental),
        "retry_incremental_coverage": round(len(retry_incremental) / len(requested), 4)
        if requested
        else 0.0,
        "duplicate_retry_count": duplicate_retries,
        "duplicate_retry_rate": round(duplicate_retries / len(retries), 4) if retries else 0.0,
        "unique_candidate_sql_ratio": round(len(unique_sql) / len(records), 4) if records else 0.0,
        "final_verified_example_count": len(final_verified),
        "final_verified_example_coverage": round(len(final_verified) / len(requested), 4)
        if requested
        else 0.0,
        "failure_buckets": dict(error_counts),
    }
    atomic_write_json(args.metrics_output, metrics)
    atomic_write_json(args.error_output, {"failure_buckets": dict(error_counts)})
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
