import datetime
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

outdir = Path("artifacts/formal_readiness")
outdir.mkdir(parents=True, exist_ok=True)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_id(db: str, q: str, prefix: str = "bird-train") -> str:
    return f"{prefix}-{db}-{sha256_text(f'{db}|{q}')[:8]}"


raw_path = Path("data/bird/raw/bird23_train_filtered.jsonl")
mini_path = Path("data/bird/raw/mini_dev_sqlite.json")
raw = [
    json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()
]
mini = json.loads(mini_path.read_text(encoding="utf-8"))

source_prov = {
    "dataset_name": "birdsql/bird23-train-filtered",
    "dataset_source": "HuggingFace birdsql/bird23-train-filtered + official BIRD train.zip",
    "dataset_revision_if_known": "birdsql/bird23-train-filtered main (HF)",
    "download_method": "official BIRD train.zip from bird-bench.oss-cn-beijing.aliyuncs.com; filtered JSON from HF mirror",
    "local_raw_path": str(raw_path),
    "row_count": len(raw),
    "file_size_bytes": raw_path.stat().st_size,
    "sha256": sha256_file(raw_path),
    "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
}
(outdir / "source_provenance.json").write_text(
    json.dumps(source_prov, ensure_ascii=False, indent=2), encoding="utf-8"
)

mini_prov = {
    "dataset_name": "birdsql/bird_mini_dev",
    "dataset_source": "HuggingFace birdsql/bird_mini_dev",
    "local_raw_path": str(mini_path),
    "row_count": len(mini),
    "file_size_bytes": mini_path.stat().st_size,
    "sha256": sha256_file(mini_path),
    "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
}
(outdir / "mini_dev_source_provenance.json").write_text(
    json.dumps(mini_prov, ensure_ascii=False, indent=2), encoding="utf-8"
)
assert len(mini) == 500, "FINAL_MINIDEV_SOURCE_MISMATCH"

train_db_ids = sorted(set(r["db_id"] for r in raw))
eval_db_ids = sorted(set(r["db_id"] for r in mini))
(outdir / "required_train_db_ids.json").write_text(
    json.dumps({"count": len(train_db_ids), "db_ids": train_db_ids}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
(outdir / "required_eval_db_ids.json").write_text(
    json.dumps({"count": len(eval_db_ids), "db_ids": eval_db_ids}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

eng = [
    json.loads(line)
    for line in Path("data/bird/splits/validation_tuning.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]
eng_keys = {(r["db_id"], r["question"].strip()) for r in eng}
eng_ids = [r["example_id"] for r in eng]
(outdir / "engineering_validation_manifest.json").write_text(
    json.dumps(
        {
            "name": "engineering_validation_tuning",
            "count": len(eng),
            "example_ids": eng_ids,
            "db_distribution": dict(Counter(r["db_id"] for r in eng)),
            "frozen": True,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

pool = [r for r in raw if (r["db_id"], r["question"].strip()) not in eng_keys]
pool_by_db = defaultdict(list)
for r in pool:
    pool_by_db[r["db_id"]].append(r)
rng = random.Random(20260818)
for db in sorted(pool_by_db):
    rng.shuffle(pool_by_db[db])
selected = []
used_keys = set()
for db in sorted(pool_by_db):
    if pool_by_db[db]:
        r = pool_by_db[db].pop(0)
        key = (r["db_id"], r["question"].strip())
        selected.append((db, r, key))
        used_keys.add(key)
while len(selected) < 256 and any(pool_by_db.values()):
    progressed = False
    for db in sorted(pool_by_db):
        if len(selected) >= 256:
            break
        if pool_by_db[db]:
            r = pool_by_db[db].pop(0)
            key = (r["db_id"], r["question"].strip())
            selected.append((db, r, key))
            used_keys.add(key)
            progressed = True
    if not progressed:
        break

formal_val_rows = []
for db, r, _key in selected:
    formal_val_rows.append(
        {
            "example_id": stable_id(db, r["question"]),
            "db_id": db,
            "question": r["question"],
            "evidence": r.get("evidence", ""),
            "gold_sql": r.get("SQL", ""),
            "split": "formal_validation",
            "source": "birdsql/bird23-train-filtered",
            "source_version": "bird23-v1",
        }
    )
formal_val_ids = [r["example_id"] for r in formal_val_rows]
Path("data/bird/splits/formal_validation.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in formal_val_rows) + "\n", encoding="utf-8"
)
formal_val_manifest = {
    "name": "formal_validation",
    "seed": 20260818,
    "algorithm_version": "db-stratified-round-robin-v1",
    "count": len(formal_val_rows),
    "example_ids": formal_val_ids,
    "db_distribution": dict(Counter(r["db_id"] for r in formal_val_rows)),
    "source_hash": source_prov["sha256"],
    "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "frozen": True,
}
(outdir / "formal_validation_manifest.json").write_text(
    json.dumps(formal_val_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)

remaining = [
    r
    for r in raw
    if (r["db_id"], r["question"].strip()) not in eng_keys
    and (r["db_id"], r["question"].strip()) not in used_keys
]
formal_train_ids = [stable_id(r["db_id"], r["question"]) for r in remaining]
formal_train_manifest = {
    "name": "formal_train",
    "count": len(remaining),
    "example_ids": formal_train_ids,
    "db_distribution": dict(Counter(r["db_id"] for r in remaining)),
    "ids_hash": sha256_text("\n".join(sorted(formal_train_ids))),
    "frozen": True,
}
(outdir / "formal_train_manifest.json").write_text(
    json.dumps(formal_train_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)

formal_split = {
    "full_filtered_train_count": len(raw),
    "engineering_validation_count": len(eng),
    "formal_validation_count": len(formal_val_rows),
    "formal_train_count": len(remaining),
    "sum_check": len(eng) + len(formal_val_rows) + len(remaining),
    "union_equals_6601": len(eng) + len(formal_val_rows) + len(remaining) == 6601,
    "overlaps": {
        "eng_vs_formal_val": len(set(eng_ids) & set(formal_val_ids)),
        "eng_vs_formal_train": len(set(eng_ids) & set(formal_train_ids)),
        "formal_val_vs_formal_train": len(set(formal_val_ids) & set(formal_train_ids)),
    },
}
(outdir / "formal_split_manifest.json").write_text(
    json.dumps(formal_split, ensure_ascii=False, indent=2), encoding="utf-8"
)

with open("artifacts/experiment/benchmark_exposure_manifest.json", encoding="utf-8") as f:
    exposure = json.load(f)
exposed = exposure.get("examples", exposure.get("exposed", []))
if exposed and isinstance(exposed[0], dict):
    exposed_ids = [e.get("question_id") or e.get("example_id") for e in exposed]
else:
    exposed_ids = list(exposed)
mini_ids = [r["question_id"] for r in mini]
mini_exposure_manifest = {
    "exposed_count": len(exposed_ids),
    "exposed_ids": exposed_ids,
    "engineering_pilot_exposed": True,
    "subset_of_final_500": set(exposed_ids) <= set(mini_ids),
}
(outdir / "mini_dev_exposure_manifest.json").write_text(
    json.dumps(mini_exposure_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
final_select500 = {
    "count": len(mini),
    "example_ids": mini_ids,
    "db_distribution": dict(Counter(r["db_id"] for r in mini)),
    "difficulty_distribution": dict(Counter(r.get("difficulty", "unknown") for r in mini)),
    "exposed_intersection": sorted(set(exposed_ids) & set(mini_ids)),
    "unexposed_count": len(set(mini_ids) - set(exposed_ids)),
}
(outdir / "final_select500_manifest.json").write_text(
    json.dumps(final_select500, ensure_ascii=False, indent=2), encoding="utf-8"
)

formal_train_keys = {(r["db_id"], r["question"].strip()) for r in remaining}
formal_val_keys = {(r["db_id"], r["question"].strip()) for r in formal_val_rows}
mini_keys = {(r["db_id"], r["question"].strip()) for r in mini}
leakage = {
    "formal_train_vs_engineering": len(formal_train_keys & eng_keys),
    "formal_train_vs_formal_validation": len(formal_train_keys & formal_val_keys),
    "formal_train_vs_mini_dev": len(formal_train_keys & mini_keys),
    "engineering_vs_formal_validation": len(eng_keys & formal_val_keys),
    "engineering_vs_mini_dev": len(eng_keys & mini_keys),
    "formal_validation_vs_mini_dev": len(formal_val_keys & mini_keys),
    "duplicate_normalized_question_count": len(raw)
    - len({(r["db_id"], r["question"].strip()) for r in raw}),
}
(outdir / "leakage_report.json").write_text(
    json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8"
)

seen = {}
dup_records = []
for r in raw:
    key = (r["db_id"], r["question"].strip())
    if key in seen:
        dup_records.append(
            {
                "db_id": r["db_id"],
                "question": r["question"],
                "sql_a": seen[key].get("SQL", ""),
                "sql_b": r.get("SQL", ""),
                "same_sql": seen[key].get("SQL", "") == r.get("SQL", ""),
            }
        )
    else:
        seen[key] = r
(outdir / "leakage_duplicates.jsonl").write_text(
    "\n".join(json.dumps(x, ensure_ascii=False) for x in dup_records) + "\n", encoding="utf-8"
)

print("source_prov", source_prov["sha256"])
print("formal_split", formal_split)
print("leakage", leakage)
print("select500", final_select500["count"], "unexposed", final_select500["unexposed_count"])
