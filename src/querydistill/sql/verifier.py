"""Result equivalence verification.

Correctness is decided on **execution results**, never on SQL string equality:

* no ``ORDER BY`` in the gold query -> multiset (row multiset) comparison
* semantically relevant ``ORDER BY`` -> ordered comparison
* NULLs, duplicate rows, integers/floats with tolerance, strings and empty
  results are all handled explicitly
* both-empty results are never accepted unconditionally (empty-result reward
  hacking); they additionally require structural sanity checks and are reported
  with reduced confidence

Full rationale: ``docs/RESULT_EQUIVALENCE.md``.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

import sqlglot
from sqlglot import exp

from .executor import ExecutionResult


@dataclass
class VerificationResult:
    equivalent: bool  # STRICT correctness only (kept as the primary alias)
    strict_equivalent: bool
    partial_credit: bool
    kind: str
    reason: str
    candidate_row_count: int = 0
    gold_row_count: int = 0
    column_match: bool = False  # diagnostic: exact column-name equality
    column_count_match: bool = False  # what strict equivalence actually requires
    order_sensitive: bool = False
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "equivalent": self.equivalent,
            "strict_equivalent": self.strict_equivalent,
            "partial_credit": self.partial_credit,
            "kind": self.kind,
            "reason": self.reason,
            "candidate_row_count": self.candidate_row_count,
            "gold_row_count": self.gold_row_count,
            "column_match": self.column_match,
            "column_count_match": self.column_count_match,
            "order_sensitive": self.order_sensitive,
            "details": self.details,
        }


def extract_table_names(sql: str, dialect: str = "sqlite") -> list[str]:
    """Best-effort extraction of FROM/JOIN source table names from an AST."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:  # noqa: BLE001 - best effort helper
        return []
    names: list[str] = []
    for table in tree.find_all(exp.Table):
        name = table.name
        if name:
            names.append(str(name))
    return names


def has_order_by(sql: str, dialect: str = "sqlite") -> bool:
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:  # noqa: BLE001 - best effort helper
        return False
    return any(isinstance(node, exp.Order) for node in tree.walk())


def _normalize_value(value: object, tolerance: float) -> object:
    if value is None:
        return None
    if isinstance(value, bytes):
        return ("__bytes__", value)
    if isinstance(value, bool):
        return ("__bool__", int(value))
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return ("__nan__",)
        if tolerance > 0:
            try:
                quantized = Decimal(str(value)).quantize(Decimal(str(tolerance)))
            except InvalidOperation:  # pragma: no cover - defensive
                quantized = Decimal(str(value))
        else:
            quantized = Decimal(str(value))
        return ("__number__", quantized)
    if isinstance(value, str):
        return ("__str__", value)
    return ("__other__", type(value).__name__, str(value))


def _values_equal(left: object, right: object, tolerance: float) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bytes) or isinstance(right, bytes):
        return isinstance(left, bytes) and isinstance(right, bytes) and left == right
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        difference = abs(float(left) - float(right))
        scale = max(1.0, abs(float(left)), abs(float(right)))
        return difference <= tolerance * scale
    if isinstance(left, float) and math.isnan(left):
        return isinstance(right, float) and math.isnan(right)
    return left == right


class ResultEquivalenceVerifier:
    """Compare candidate vs gold execution results."""

    def __init__(self, float_tolerance: float = 1e-6):
        if float_tolerance < 0:
            raise ValueError("float_tolerance must be non-negative")
        self.float_tolerance = float(float_tolerance)

    def verify(
        self,
        candidate: ExecutionResult,
        gold: ExecutionResult,
        candidate_sql: str | None = None,
        gold_sql: str | None = None,
        schema_tables: set[str] | None = None,
    ) -> VerificationResult:
        base = VerificationResult(
            equivalent=False,
            strict_equivalent=False,
            partial_credit=False,
            kind="not_equivalent",
            reason="",
            candidate_row_count=candidate.row_count,
            gold_row_count=gold.row_count,
            column_match=candidate.columns == gold.columns,
            column_count_match=len(candidate.columns) == len(gold.columns),
        )

        if not gold.success:
            base.reason = "gold query did not execute successfully; cannot verify"
            return base
        if not candidate.success:
            base.reason = (
                f"candidate query failed ({candidate.error_type}): {candidate.error_message}"
            )
            return base

        # Truncated results are never strict-equivalent (no streaming/hash
        # equivalence is implemented in this round).
        if candidate.truncated or gold.truncated:
            base.kind = "truncated_unverified"
            base.reason = (
                "result comparison is truncated "
                f"(candidate_truncated={candidate.truncated}, "
                f"gold_truncated={gold.truncated}); strict equivalence cannot be proven"
            )
            return base

        if len(candidate.columns) != len(gold.columns):
            base.reason = (
                "column count mismatch: "
                f"candidate {len(candidate.columns)} vs gold {len(gold.columns)}; "
                "column names are only a secondary diagnostic"
            )
            return base

        order_sensitive = bool(gold_sql) and has_order_by(gold_sql or "")
        base.order_sensitive = order_sensitive

        if candidate.row_count == 0 and gold.row_count == 0:
            return self._verify_both_empty(base, candidate_sql, gold_sql, schema_tables)

        if order_sensitive:
            if candidate.row_count != gold.row_count:
                base.reason = (
                    "gold is ORDER BY sensitive and row counts differ "
                    f"({candidate.row_count} vs {gold.row_count})"
                )
                return base
            for index, (candidate_row, gold_row) in enumerate(
                zip(candidate.rows, gold.rows, strict=True)
            ):
                if len(candidate_row) != len(gold_row):
                    base.reason = f"row {index} width differs"
                    return base
                for col, (candidate_value, gold_value) in enumerate(
                    zip(candidate_row, gold_row, strict=True)
                ):
                    if not _values_equal(candidate_value, gold_value, self.float_tolerance):
                        base.reason = (
                            f"ordered mismatch at row {index} col {col}: "
                            f"{candidate_value!r} vs {gold_value!r}"
                        )
                        return base
            base.equivalent = True
            base.strict_equivalent = True
            base.kind = "ordered_rows"
            base.reason = "candidate rows match gold rows in ORDER BY-sensitive order"
            return base

        candidate_bag = Counter(
            tuple(_normalize_value(value, self.float_tolerance) for value in row)
            for row in candidate.rows
        )
        gold_bag = Counter(
            tuple(_normalize_value(value, self.float_tolerance) for value in row)
            for row in gold.rows
        )
        if candidate_bag == gold_bag:
            base.equivalent = True
            base.strict_equivalent = True
            base.kind = "unordered_rows"
            base.reason = "candidate rows match gold rows as multisets (no ORDER BY)"
        else:
            base.reason = "row multisets differ"
        return base

    def _verify_both_empty(
        self,
        base: VerificationResult,
        candidate_sql: str | None,
        gold_sql: str | None,
        schema_tables: set[str] | None,
    ) -> VerificationResult:
        candidate_tables = set(extract_table_names(candidate_sql or ""))
        gold_tables = set(extract_table_names(gold_sql or ""))

        problems: list[str] = []
        if not candidate_sql:
            problems.append("candidate SQL unavailable for structural check")
        if not candidate_tables:
            problems.append("candidate has no FROM source (constant / schema-independent query)")
        if gold_tables and not candidate_tables.issubset(gold_tables):
            problems.append(
                f"candidate tables {sorted(candidate_tables)} are not a subset of "
                f"gold tables {sorted(gold_tables)}"
            )
        if schema_tables is not None and candidate_tables:
            unknown = candidate_tables - schema_tables
            if unknown:
                problems.append(f"candidate references unknown tables {sorted(unknown)}")

        if problems:
            base.kind = "empty_unverified"
            base.reason = "both results empty but structural sanity failed: " + "; ".join(problems)
            base.details["problems"] = problems
            return base

        # Both-empty results are NEVER strict correctness. They may receive a
        # small shaping/partial credit in GRPO, but never count as execution
        # accuracy and never enter a teacher-verified dataset.
        base.partial_credit = True
        base.kind = "empty_structural_partial"
        base.reason = (
            "both results empty; structural sanity passed (candidate tables subset of "
            "gold tables) but this is only PARTIAL credit, not strict correctness"
        )
        base.details["candidate_tables"] = sorted(candidate_tables)
        base.details["gold_tables"] = sorted(gold_tables)
        return base
