"""Composite reward orchestrator (round-2 hardened).

* ``score_once`` computes breakdown + parse + safety + candidate execution +
  verification + trace in ONE pass; logging never re-executes SQL.
* Gold results are cached per dataset hash / db_id / example_id / gold_sql_hash
  and invalidate automatically when the dataset hash changes.
* Unsafe SQL is never executed and the total is pinned to -1.0.
* Only STRICT verification earns the full correctness reward; empty-partial
  results receive shaping credit only.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..outputs.fallback import extract_fallback_sql
from ..outputs.parser import ParseResult, parse_model_output
from ..sql.environment import SQLExecutionEnvironment
from ..sql.executor import ExecutionResult
from ..sql.safety import SafetyDecision, validate_sql
from ..sql.verifier import ResultEquivalenceVerifier, VerificationResult
from ..utils import atomic_write_json, load_json, utc_now_iso
from .base import RewardBreakdown
from .correctness_reward import correctness_reward
from .execution_reward import execution_reward
from .format_reward import format_reward
from .parse_reward import parse_reward
from .safety_reward import safety_reward


class GoldResultCache:
    """Persistent read cache of gold execution results.

    Key = dataset_hash + database_fingerprint + db_id + example_id +
    gold_sql_hash. A database/source hash change produces a different key, so
    stale entries are never reused.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        dataset_hash: str,
        database_fingerprints: dict[str, str] | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.dataset_hash = dataset_hash
        self.database_fingerprints = database_fingerprints or {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _path(self, db_id: str, example_id: str, gold_sql: str) -> Path:
        sql_hash = hashlib.sha256(gold_sql.encode("utf-8")).hexdigest()[:16]
        db_fingerprint = self.database_fingerprints.get(db_id, "unknown")[:16]
        safe_db = "".join(c for c in db_id if c.isalnum() or c in "-_")
        safe_example = "".join(c for c in example_id if c.isalnum() or c in "._-")
        return (
            self.cache_dir
            / self.dataset_hash[:16]
            / db_fingerprint
            / f"{safe_db}_{safe_example}_{sql_hash}.json"
        )

    def get(self, db_id: str, example_id: str, gold_sql: str) -> ExecutionResult | None:
        path = self._path(db_id, example_id, gold_sql)
        if not path.exists():
            self.misses += 1
            return None
        try:
            payload = load_json(path)
            self.hits += 1
            return ExecutionResult(**payload)
        except Exception:  # noqa: BLE001 - corrupted cache entry is treated as miss
            self.misses += 1
            return None

    def put(self, db_id: str, example_id: str, gold_sql: str, result: ExecutionResult) -> None:
        path = self._path(db_id, example_id, gold_sql)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(path, result.as_dict())

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses}


@dataclass
class RewardTrace:
    breakdown: RewardBreakdown
    parse: ParseResult
    safety: SafetyDecision | None
    candidate_execution: ExecutionResult | None
    gold_execution: ExecutionResult | None
    verification: VerificationResult | None
    candidate_executions: int
    gold_cache_stats: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now_iso)

    def as_dict(self) -> dict:
        return {
            "reward_breakdown": self.breakdown.as_dict(),
            "parse": self.parse.as_dict(),
            "safety": self.safety.as_dict() if self.safety else None,
            "candidate_execution": (
                self.candidate_execution.as_dict() if self.candidate_execution else None
            ),
            "gold_execution": self.gold_execution.as_dict() if self.gold_execution else None,
            "verification": self.verification.as_dict() if self.verification else None,
            "candidate_executions": self.candidate_executions,
            "gold_cache_stats": self.gold_cache_stats,
            "timestamp": self.timestamp,
        }


class CompositeReward:
    def __init__(
        self,
        environment: SQLExecutionEnvironment,
        verifier: ResultEquivalenceVerifier | None = None,
        require_plan: bool = False,
        gold_cache: GoldResultCache | None = None,
    ):
        self.environment = environment
        self.verifier = verifier or ResultEquivalenceVerifier()
        self.require_plan = require_plan
        self.gold_cache = gold_cache
        self._schema_tables: dict[str, set[str]] = {}
        self.execution_count = 0  # instrumentation for tests

    def schema_tables(self, db_id: str) -> set[str]:
        if db_id not in self._schema_tables:
            self._schema_tables[db_id] = set(self.environment.table_names(db_id))
        return self._schema_tables[db_id]

    def _field(self, example: Any, name: str) -> str:
        return str(getattr(example, name, None) or example[name])

    def score_once(self, example: Any, model_output: str) -> RewardTrace:
        db_id = self._field(example, "db_id")
        gold_sql = self._field(example, "gold_sql")

        parsed = parse_model_output(model_output, require_plan=self.require_plan)
        fallback_parsed = None
        if parsed.sql is None:
            fallback_sql = extract_fallback_sql(model_output)
            if fallback_sql:
                fallback_parsed = parse_model_output(
                    f"<sql>\n{fallback_sql}\n</sql>", require_plan=self.require_plan
                )
        effective_parsed = parsed if parsed.sql else fallback_parsed
        format_score, format_notes = format_reward(parsed, require_plan=self.require_plan)
        parse_score, parse_notes = parse_reward(effective_parsed if effective_parsed else parsed)
        if fallback_parsed is not None and fallback_parsed.sql:
            parse_notes = {**parse_notes, "fallback_sql_used": True}

        decision = (
            validate_sql(effective_parsed.sql)
            if effective_parsed is not None and effective_parsed.sql
            else None
        )
        safety_score, safety_notes = safety_reward(decision)

        candidate_result = None
        gold_result = None
        verification = None
        candidate_executions = 0

        if (
            decision is not None
            and decision.safe
            and effective_parsed is not None
            and effective_parsed.sql is not None
        ):
            # Candidate executes exactly once here.
            self.execution_count += 1
            candidate_executions = 1
            candidate_result = self.environment.execute(db_id, effective_parsed.sql)
            gold_result = self._gold_result(example, db_id, gold_sql)
            execution_score, execution_notes = execution_reward(candidate_result)
            verification = self.verifier.verify(
                candidate=candidate_result,
                gold=gold_result,
                candidate_sql=effective_parsed.sql,
                gold_sql=gold_sql,
                schema_tables=self.schema_tables(db_id),
            )
            correctness_score, correctness_notes = correctness_reward(verification)
        else:
            execution_score, execution_notes = execution_reward(None)
            correctness_score, correctness_notes = correctness_reward(None)

        if decision is not None and not decision.safe:
            total = -1.0
        else:
            total = max(
                -1.0,
                format_score + parse_score + safety_score + execution_score + correctness_score,
            )

        notes: dict[str, str] = {}
        for prefix, source in (
            ("format", format_notes),
            ("parse", parse_notes),
            ("safety", safety_notes),
            ("execution", execution_notes),
            ("correctness", correctness_notes),
        ):
            for key, value in source.items():
                notes[f"{prefix}_{key}"] = str(value)

        return RewardTrace(
            breakdown=RewardBreakdown(
                format=format_score,
                parse=parse_score,
                safety=safety_score,
                execution=execution_score,
                correctness=correctness_score,
                total=round(total, 6),
                notes=notes,
            ),
            parse=parsed,
            safety=decision,
            candidate_execution=candidate_result,
            gold_execution=gold_result,
            verification=verification,
            candidate_executions=candidate_executions,
            gold_cache_stats=self.gold_cache.stats() if self.gold_cache else {},
        )

    def _gold_result(self, example: Any, db_id: str, gold_sql: str) -> ExecutionResult:
        example_id = self._field(example, "example_id")
        if self.gold_cache is not None:
            cached = self.gold_cache.get(db_id, example_id, gold_sql)
            if cached is not None:
                return cached
        result = self.environment.execute(db_id, gold_sql)
        if self.gold_cache is not None:
            self.gold_cache.put(db_id, example_id, gold_sql, result)
        return result

    def evaluate(self, example: Any, model_output: str) -> RewardBreakdown:
        return self.score_once(example, model_output).breakdown

    def evaluate_trace(self, example: Any, model_output: str) -> dict:
        return self.score_once(example, model_output).as_dict()
