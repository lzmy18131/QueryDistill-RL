#!/usr/bin/env python3
"""Build paired Gold/Distilled manifest from diagnostic + collection teacher candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from querydistill.data.paired import build_paired_targets
from querydistill.data.schema import load_distillation_records, load_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", required=True)
    parser.add_argument("--diagnostic-attempts", required=True)
    parser.add_argument("--collection-candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verified-output", required=True)
    args = parser.parse_args()

    diag = load_distillation_records(args.diagnostic_attempts)
    coll = load_distillation_records(args.collection_candidates)
    records = diag + coll
    verified = [r for r in records if r.execution_equivalent]

    with Path(args.verified_output).open("w", encoding="utf-8") as f:
        for r in verified:
            f.write(json.dumps(r.model_dump(), ensure_ascii=False) + "\n")

    examples = load_examples(args.examples)
    paired = build_paired_targets(
        examples,
        records,
        examples_path=args.examples,
        require_all=False,
        selection_policy="min_candidate_index",
    )
    paired.write_manifest(args.output)
    print(
        json.dumps(
            {
                "requested_count": paired.requested_count,
                "paired_count": paired.paired_count,
                "verified_teacher_coverage": paired.verified_teacher_coverage,
                "db_distribution": {
                    db: sum(
                        1
                        for e in examples
                        if e.example_id in set(paired.example_ids) and e.db_id == db
                    )
                    for db in sorted(
                        set(e.db_id for e in examples if e.example_id in set(paired.example_ids))
                    )
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
