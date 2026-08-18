"""Deterministic train_core / validation_tuning splitter.

The split is stratified by ``db_id`` using round-robin selection after a seeded
shuffle, so it is representative across databases and never uses first-N.
``validation_tuning`` is removed from train_core and is forbidden in every
training path by :class:`~querydistill.data.split_policy.SplitPolicy`.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from ..utils import atomic_write_json, load_jsonl, sha256_file, utc_now_iso
from .schema import Example, load_examples


def _example_to_dict(example: Example) -> dict:
    return example.model_dump()


def create_train_validation_split(
    examples_path: str | Path,
    output_dir: str | Path,
    validation_size: int = 120,
    seed: int = 3407,
) -> dict:
    examples_path = Path(examples_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = [e for e in load_examples(examples_path) if e.split == "train"]
    if len(examples) <= validation_size:
        raise ValueError(
            f"cannot create validation_tuning of size {validation_size} from "
            f"{len(examples)} train examples"
        )

    rng = random.Random(seed)
    by_db: dict[str, list[Example]] = {}
    for example in examples:
        by_db.setdefault(example.db_id, []).append(example)
    for db_examples in by_db.values():
        rng.shuffle(db_examples)

    validation: list[Example] = []
    seen: set[str] = set()
    # Round-robin across DBs until the validation budget is filled.
    db_order = sorted(by_db)
    exhausted = set()
    while len(validation) < validation_size and len(exhausted) < len(db_order):
        for db_id in db_order:
            if len(validation) >= validation_size:
                break
            if db_id in exhausted:
                continue
            candidates = [e for e in by_db[db_id] if e.example_id not in seen]
            if not candidates:
                exhausted.add(db_id)
                continue
            chosen = candidates[0]
            seen.add(chosen.example_id)
            validation.append(chosen)
            by_db[db_id] = [e for e in by_db[db_id] if e.example_id != chosen.example_id]

    validation_ids = [e.example_id for e in validation]
    validation_id_set = set(validation_ids)
    train_core = [e for e in examples if e.example_id not in validation_id_set]

    # Mark validation_tuning split.
    validation_tuning = [e.model_copy(update={"split": "validation_tuning"}) for e in validation]

    train_core_path = output_dir / "train_core.jsonl"
    validation_tuning_path = output_dir / "validation_tuning.jsonl"
    with train_core_path.open("w", encoding="utf-8") as f:
        for e in train_core:
            f.write(json.dumps(_example_to_dict(e), ensure_ascii=False) + "\n")
    with validation_tuning_path.open("w", encoding="utf-8") as f:
        for e in validation_tuning:
            f.write(json.dumps(_example_to_dict(e), ensure_ascii=False) + "\n")

    def _db_distribution(items: list[Example]) -> dict[str, int]:
        dist: dict[str, int] = {}
        for e in items:
            dist[e.db_id] = dist.get(e.db_id, 0) + 1
        return dict(sorted(dist.items()))

    manifest = {
        "source_path": str(examples_path.resolve()),
        "source_hash": sha256_file(examples_path),
        "seed": seed,
        "validation_size": validation_size,
        "train_core_count": len(train_core),
        "validation_tuning_count": len(validation_tuning),
        "train_core_ids_hash": hashlib.sha256(
            "\n".join(sorted(e.example_id for e in train_core)).encode()
        ).hexdigest(),
        "validation_ids": validation_ids,
        "train_core_db_distribution": _db_distribution(train_core),
        "validation_db_distribution": _db_distribution(validation_tuning),
        "created_at": utc_now_iso(),
    }
    manifest_path = output_dir / "split_manifest.json"
    atomic_write_json(manifest_path, manifest)

    return {
        "train_core_path": str(train_core_path),
        "validation_tuning_path": str(validation_tuning_path),
        "manifest_path": str(manifest_path),
        **manifest,
    }
