"""Official BIRD data adapter.

Reads `birdsql/bird23-train-filtered` and `birdsql/bird_mini_dev` records and
converts them into the project ``Example`` schema with SQLite db registry.

The raw BIRD JSON/JSONL files are kept on D: under ``data/bird/raw`` and the
SQLite databases under ``data/bird/{train,mini_dev}/databases``. The adapter
never copies the full BIRD dataset into Git.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from ..sql.executor import SafeSQLExecutor
from ..utils import atomic_write_json, atomic_write_text
from .schema import Example


@dataclass
class BirdPrepareReport:
    train_count: int = 0
    dev_count: int = 0
    test_count: int = 0
    database_count: int = 0
    skipped_missing_db: int = 0
    skipped_gold_failure: int = 0
    split_counts: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "train_count": self.train_count,
            "dev_count": self.dev_count,
            "test_count": self.test_count,
            "database_count": self.database_count,
            "skipped_missing_db": self.skipped_missing_db,
            "skipped_gold_failure": self.skipped_gold_failure,
            "split_counts": self.split_counts,
        }


def schema_for_db(db_path: Path) -> str:
    """Return a deterministic schema text from SQLite ``sqlite_master``."""
    connection = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type IN ('table','view') AND sql IS NOT NULL "
            "ORDER BY name"
        ).fetchall()
    finally:
        connection.close()
    return "\n".join(row[0].strip() for row in rows if row[0].strip())


def _gold_ok(db_path: Path, sql: str) -> bool:
    """Check a gold SELECT executes under the project executor limits."""
    try:
        result = SafeSQLExecutor(db_path, max_rows=1000, max_execution_ms=3000).execute(sql)
        return result.success
    except Exception:
        return False


def load_bird_train_filtered(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_bird_mini_dev(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _bird_identity_payload(record: dict) -> dict:
    """Canonical identity tuple for a BIRD example.

    The raw filtered train source has no official ``question_id``, so the stable
    ID must be derived from content that uniquely identifies the row.  We use a
    canonical JSON object (sorted keys, UTF-8, no Python ``repr`` dependence)
    rather than the built-in ``hash()``.
    """
    return {
        "db_id": str(record["db_id"]),
        "question": str(record["question"]),
        "evidence": str(record.get("evidence") or "").strip(),
        "gold_sql": str(record.get("SQL") or record.get("gold_sql") or "").strip(),
    }


def canonical_bird_example_id(record: dict, prefix: str = "bird-train") -> str:
    """Return the stable canonical BIRD example ID.

    If the source record provides an official stable ``question_id`` (Mini-Dev),
    it is used verbatim.  Otherwise a SHA-256 digest of the canonical identity
    payload is used; this is deterministic across processes and
    ``PYTHONHASHSEED`` values.
    """
    question_id = record.get("question_id")
    if question_id is not None:
        return f"{prefix}-{question_id}"
    canonical = json.dumps(
        _bird_identity_payload(record),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{prefix}-{digest}"


def bird_content_signature(record: dict) -> str:
    """SHA-256 of the canonical identity payload (used for ID migration)."""
    canonical = json.dumps(
        _bird_identity_payload(record),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _bird_record_to_example(
    record: dict,
    db_paths: dict[str, Path],
    split: str,
    source: str,
    source_version: str,
    example_id_prefix: str,
) -> Example | None:
    db_id = str(record["db_id"])
    db_path = db_paths.get(db_id)
    if db_path is None:
        return None
    schema_text = schema_for_db(db_path)
    question = str(record["question"]).strip()
    sql = str(record.get("SQL") or record.get("gold_sql") or "").strip()
    if not question or not sql or not schema_text:
        return None
    example_id = canonical_bird_example_id(record, prefix=example_id_prefix)
    return Example(
        example_id=example_id,
        db_id=db_id,
        question=question,
        schema_text=schema_text,
        gold_sql=sql,
        split=split,
        source=source,
        source_version=source_version,
        evidence=str(record.get("evidence") or ""),
    )


def build_bird_examples(
    train_path: str | Path,
    mini_dev_path: str | Path,
    train_db_dir: str | Path,
    mini_dev_db_dir: str | Path,
    max_train: int | None = 50,
    max_dev: int | None = 20,
) -> tuple[list[Example], BirdPrepareReport]:
    train_db_dir = Path(train_db_dir)
    mini_dev_db_dir = Path(mini_dev_db_dir)
    train_dbs = {p.stem: p for p in train_db_dir.glob("*.sqlite")}
    mini_dbs = {p.stem: p for p in mini_dev_db_dir.glob("*.sqlite")}

    report = BirdPrepareReport()
    examples: list[Example] = []

    train_records = load_bird_train_filtered(train_path)
    train_used = 0
    for record in train_records:
        if max_train is not None and train_used >= max_train:
            break
        example = _bird_record_to_example(
            record,
            train_dbs,
            split="train",
            source="birdsql/bird23-train-filtered",
            source_version="bird23-v1",
            example_id_prefix="bird-train",
        )
        if example is None:
            report.skipped_missing_db += 1
            continue
        if not _gold_ok(train_dbs[example.db_id], example.gold_sql):
            report.skipped_gold_failure += 1
            continue
        examples.append(example)
        train_used += 1
    report.train_count = train_used

    mini_records = load_bird_mini_dev(mini_dev_path)
    dev_used = 0
    for record in mini_records:
        if max_dev is not None and dev_used >= max_dev:
            break
        example = _bird_record_to_example(
            record,
            mini_dbs,
            split="dev",
            source="birdsql/bird_mini_dev",
            source_version="bird23-minidev-v1",
            example_id_prefix="bird-dev",
        )
        if example is None:
            report.skipped_missing_db += 1
            continue
        if not _gold_ok(mini_dbs[example.db_id], example.gold_sql):
            report.skipped_gold_failure += 1
            continue
        examples.append(example)
        dev_used += 1
    report.dev_count = dev_used

    report.database_count = len(set(e.db_id for e in examples))
    report.split_counts = {
        split: sum(1 for e in examples if e.split == split) for split in ("train", "dev", "test")
    }
    return examples, report


def write_bird_registry(
    registry_path: str | Path,
    train_db_dir: str | Path,
    mini_dev_db_dir: str | Path,
) -> None:
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    databases: dict[str, str] = {}
    for db_dir in (Path(train_db_dir), Path(mini_dev_db_dir)):
        for db_file in db_dir.glob("*.sqlite"):
            rel = db_file.resolve().relative_to(registry_path.parent.resolve())
            databases[db_file.stem] = rel.as_posix()
    atomic_write_json(registry_path, {"databases": databases})


def write_bird_examples(examples: list[Example], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for split in ("train", "dev", "test"):
        split_examples = [e for e in examples if e.split == split]
        if not split_examples:
            continue
        path = output_dir / f"{split}.jsonl"
        lines = (
            "\n".join(json.dumps(e.model_dump(), ensure_ascii=False) for e in split_examples) + "\n"
        )
        atomic_write_text(path, lines)
        written[split] = path
    return written
