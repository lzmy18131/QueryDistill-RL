"""Dataset schema (pydantic) and loaders.

Every record in the project carries the fields required by the project spec
(example_id, db_id, question, schema_text, gold_sql, split, source,
source_version; distillation records additionally carry teacher metadata and
verification results).
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from ..utils import load_jsonl

ALLOWED_SPLITS = frozenset({"train", "dev", "test", "calibration", "validation_tuning"})
_DB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_EXAMPLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class Example(BaseModel):
    example_id: str
    db_id: str
    question: str = Field(min_length=1)
    schema_text: str = Field(min_length=1)
    gold_sql: str = Field(min_length=1)
    split: str
    source: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    evidence: str = ""

    @field_validator("split")
    @classmethod
    def _split_allowed(cls, value: str) -> str:
        if value not in ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {sorted(ALLOWED_SPLITS)}, got {value!r}")
        return value

    @field_validator("db_id")
    @classmethod
    def _db_id_shape(cls, value: str) -> str:
        if not _DB_ID_RE.match(value):
            raise ValueError(f"invalid db_id {value!r}")
        return value

    @field_validator("example_id")
    @classmethod
    def _example_id_shape(cls, value: str) -> str:
        if not _EXAMPLE_ID_RE.match(value):
            raise ValueError(f"invalid example_id {value!r}")
        return value


class DistillationRecord(BaseModel):
    example_id: str
    teacher_model: str
    teacher_model_revision: str = "unknown"
    teacher_prompt_version: str
    candidate_index: int = Field(ge=0)
    raw_candidate_output: str = Field(min_length=1)
    candidate_sql: str | None = None
    candidate_plan: str | None = None
    parse_valid: bool
    safe: bool
    execution_success: bool
    execution_equivalent: bool
    generation_config: dict = Field(default_factory=dict)
    created_at: str
    attempt_index: int = 0
    generation_seed: int | None = None
    normalized_sql_hash: str | None = None
    safety_error_type: str | None = None
    execution_error_type: str | None = None
    execution_error_message_sanitized: str | None = None
    verification_kind: str | None = None
    retry_reason: str | None = None


class DuplicateExampleError(ValueError):
    pass


def migrate_legacy_distillation_record(record: dict) -> dict:
    """Convert round-1 records (whole protocol text in ``candidate_sql``) to the
    corrected parsed-SQL schema."""
    if "raw_candidate_output" in record:
        return record
    raw = str(record.get("candidate_sql") or "")
    from ..outputs.parser import parse_model_output

    parsed = parse_model_output(raw) if raw else None
    record["raw_candidate_output"] = raw
    record["candidate_sql"] = parsed.sql if parsed else None
    record["candidate_plan"] = parsed.plan if parsed else None
    if parsed is not None:
        record["parse_valid"] = bool(parsed.sql) and not parsed.parse_error
    return record


def load_examples(path: str | Path, declared_split: str | None = None) -> list[Example]:
    """Load examples with duplicate-id detection and optional split-mismatch guard."""
    records = load_jsonl(Path(path))
    examples: list[Example] = []
    seen: set[str] = set()
    for record in records:
        example = Example.model_validate(record)
        if declared_split is not None and example.split != declared_split:
            raise ValueError(
                f"split mismatch in {path}: file declared {declared_split!r} but "
                f"{example.example_id} has {example.split!r}"
            )
        if example.example_id in seen:
            raise DuplicateExampleError(f"duplicate example_id {example.example_id!r} in {path}")
        seen.add(example.example_id)
        examples.append(example)
    return examples


def load_distillation_records(path: str | Path) -> list[DistillationRecord]:
    records = load_jsonl(Path(path))
    return [
        DistillationRecord.model_validate(migrate_legacy_distillation_record(record))
        for record in records
    ]
