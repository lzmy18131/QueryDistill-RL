"""Evaluation metrics aggregation (round-2 semantics)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .errors import ErrorBucket


@dataclass
class EvaluationRecord:
    example_id: str
    split: str
    db_id: str
    sql: str | None
    format_ok: bool
    sql_parse_ok: bool
    parse_ok: bool
    safe: bool
    execution_success: bool
    execution_equivalent: bool  # STRICT only
    verification_partial: bool
    verification_kind: str
    error_bucket: str
    latency_ms: float
    exact_match: bool
    safety_error_type: str = "none"
    execution_error_type: str = "none"
    row_count: int = 0
    gold_row_count: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationMetrics:
    records: list[EvaluationRecord] = field(default_factory=list)

    def aggregate(self) -> dict:
        total = len(self.records)
        strict_correct = sum(1 for r in self.records if r.execution_equivalent)
        partial = sum(1 for r in self.records if r.verification_partial)
        executed = sum(1 for r in self.records if r.execution_success)
        unsafe = sum(1 for r in self.records if r.error_bucket == ErrorBucket.UNSAFE_SQL.value)
        exact = sum(1 for r in self.records if r.exact_match)
        sql_parse_ok = sum(1 for r in self.records if r.sql_parse_ok)
        format_ok = sum(1 for r in self.records if r.format_ok)
        latencies = [r.latency_ms for r in self.records]

        buckets: dict[str, int] = {}
        for record in self.records:
            buckets[record.error_bucket] = buckets.get(record.error_bucket, 0) + 1

        return {
            "num_examples": total,
            "execution_accuracy": strict_correct / total if total else 0.0,
            "format_valid_rate": format_ok / total if total else 0.0,
            "sql_parse_valid_rate": sql_parse_ok / total if total else 0.0,
            "valid_sql_rate": sql_parse_ok / total if total else 0.0,
            "execution_success_rate": executed / total if total else 0.0,
            "unsafe_sql_rate": unsafe / total if total else 0.0,
            "exact_match_secondary": exact / total if total else 0.0,
            "partial_equivalence_rate": partial / total if total else 0.0,
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "error_buckets": buckets,
            "per_split": self._per_split(),
        }

    def _per_split(self) -> dict:
        output: dict[str, dict] = {}
        for split in sorted({r.split for r in self.records}):
            subset = [r for r in self.records if r.split == split]
            output[split] = {
                "num_examples": len(subset),
                "execution_accuracy": (
                    sum(1 for r in subset if r.execution_equivalent) / len(subset)
                    if subset
                    else 0.0
                ),
            }
        return output
