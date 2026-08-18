#!/usr/bin/env python3
"""Build REAL BIRD teacher pilot metrics and paired Gold/Distilled manifest.

This is a pilot bookkeeping tool. It does not modify the training/evaluation
framework or add new algorithms; it consumes real teacher candidates and writes
the artifact files required by the Phase 1 protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from querydistill.data.paired import build_paired_targets
from querydistill.data.schema import load_distillation_records, load_examples
from querydistill.utils import atomic_write_json, load_json, utc_now_iso


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--pilot-ids", required=True)
    parser.add_argument("--verified-output", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--paired-output", required=True)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    examples_path = Path(args.examples)
    pilot_ids = load_json(args.pilot_ids)
    manifest = load_json(args.manifest) if args.manifest and Path(args.manifest).exists() else {}
    records = load_distillation_records(candidates_path)
    requested_ids = list(pilot_ids["example_ids"])
    requested = set(requested_ids)

    all_records = [r for r in records if r.example_id in requested]
    generated = len(all_records)
    parse_valid = sum(1 for r in all_records if r.parse_valid and r.candidate_sql)
    safe = sum(1 for r in all_records if r.safe)
    execution_success = sum(1 for r in all_records if r.execution_success)
    strict_verified = sum(1 for r in all_records if r.execution_equivalent)
    verified_example_ids = sorted({r.example_id for r in all_records if r.execution_equivalent})
    teacher_examples = len({r.example_id for r in all_records})

    metrics = {
        "teacher_model": manifest.get("teacher_model") or pilot_ids.get("teacher_model"),
        "teacher_model_revision": manifest.get("teacher_model_revision")
        or pilot_ids.get("teacher_model_revision"),
        "teacher_prompt_version": manifest.get("teacher_prompt_version")
        or pilot_ids.get("teacher_prompt_version"),
        "generation_config": manifest.get("generation_config")
        or pilot_ids.get("generation_config"),
        "requested_count": len(requested_ids),
        "teacher_examples": teacher_examples,
        "generated_candidates": generated,
        "mean_candidates_per_example": round(generated / teacher_examples, 4)
        if teacher_examples
        else 0.0,
        "parse_valid_rate": round(parse_valid / generated, 4) if generated else 0.0,
        "safe_sql_rate": round(safe / generated, 4) if generated else 0.0,
        "execution_success_rate": round(execution_success / generated, 4) if generated else 0.0,
        "strict_verified_rate": round(strict_verified / generated, 4) if generated else 0.0,
        "verified_example_count": len(verified_example_ids),
        "verified_example_coverage": round(len(verified_example_ids) / len(requested_ids), 4)
        if requested_ids
        else 0.0,
        "verified_example_ids": verified_example_ids,
        "created_at": utc_now_iso(),
    }

    Path(args.verified_output).parent.mkdir(parents=True, exist_ok=True)
    verified_records = [r for r in all_records if r.execution_equivalent]
    with Path(args.verified_output).open("w", encoding="utf-8") as handle:
        for record in verified_records:
            handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")

    atomic_write_json(Path(args.metrics_output), metrics)

    examples = load_examples(examples_path)
    paired = build_paired_targets(
        examples, all_records, examples_path=examples_path, require_all=False
    )
    paired.write_manifest(args.paired_output)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("paired manifest:", args.paired_output)
    print("paired_count:", paired.paired_count)
    print("dropped_missing_teacher:", paired.dropped_missing_teacher)


if __name__ == "__main__":
    main()
