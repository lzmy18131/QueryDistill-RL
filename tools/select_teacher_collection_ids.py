#!/usr/bin/env python3
"""Select additional train_core examples for expanded teacher collection.

Excludes examples already used in teacher_diagnostic. Stratified round-robin
across DBs, deterministic.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from querydistill.data.schema import load_examples
from querydistill.utils import atomic_write_json, load_json, sha256_file, utc_now_iso


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-core", required=True)
    parser.add_argument("--diagnostic-ids", required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    examples = [e for e in load_examples(args.train_core) if e.split == "train"]
    diag = load_json(args.diagnostic_ids)
    excluded = {e["example_id"] for e in diag["examples"]}
    pool = [e for e in examples if e.example_id not in excluded]

    by_db = {}
    for e in pool:
        by_db.setdefault(e.db_id, []).append(e)
    rng = random.Random(args.seed)
    for v in by_db.values():
        rng.shuffle(v)

    selected = []
    used = set()
    db_order = sorted(by_db)
    exhausted = set()
    while len(selected) < args.count and len(exhausted) < len(db_order):
        for db in db_order:
            if len(selected) >= args.count:
                break
            if db in exhausted:
                continue
            cand = [e for e in by_db[db] if e.example_id not in used]
            if not cand:
                exhausted.add(db)
                continue
            e = cand[0]
            used.add(e.example_id)
            selected.append(e)
            by_db[db] = [x for x in by_db[db] if x.example_id != e.example_id]

    payload = {
        "selection_policy": "stratified_round_robin_train_core_excluding_diagnostic",
        "seed": args.seed,
        "count": len(selected),
        "db_distribution": {
            db: sum(1 for e in selected if e.db_id == db)
            for db in sorted(set(e.db_id for e in selected))
        },
        "examples": [
            {"example_id": e.example_id, "db_id": e.db_id, "selection_order": i + 1}
            for i, e in enumerate(selected)
        ],
        "train_core_source_hash": sha256_file(args.train_core),
        "created_at": utc_now_iso(),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, payload)
    print(json.dumps(payload["db_distribution"], indent=2))
    print("count", payload["count"])


if __name__ == "__main__":
    main()
