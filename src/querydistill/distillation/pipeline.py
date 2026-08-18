"""Resumable teacher distillation pipeline with adaptive retry (signal recovery).

Flow per TRAIN example (dev/test/validation_tuning excluded by SplitPolicy):

example -> policy prompt -> Teacher -> raw candidate -> parse ->
safety -> execute parsed candidate_sql + cached gold -> STRICT result
equivalence -> verified record

Adaptive retry:
* attempt 1 is always generated;
* if the candidate is not strictly verified, attempt 2 may be generated with
  constraint-based feedback (never gold SQL/results);
* generation seeds are deterministically derived from global_seed +
  example_id + attempt_index;
* progress counts examples and candidates separately.
"""

from __future__ import annotations

import contextlib
import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ..data.schema import DistillationRecord, load_examples
from ..data.split_policy import SplitPolicy, assert_training_splits
from ..outputs.parser import parse_model_output
from ..outputs.prompting import build_prompt
from ..sql.environment import SQLExecutionEnvironment
from ..sql.safety import SafetyDecision, validate_sql
from ..sql.verifier import ResultEquivalenceVerifier
from ..utils import append_jsonl, atomic_write_json, load_json, load_jsonl, sha256_file, utc_now_iso
from .backends import MockTeacherBackend, TeacherBackend, TransformersTeacherBackend

_SAFE_CONTEXT_KEYS = frozenset({"example_id", "db_id", "split"})

RETRY_FEEDBACK_MULTI = (
    "Previous answer violated the single read-only SQLite statement constraint. "
    "Return exactly one SELECT/CTE."
)
RETRY_FEEDBACK_UNSAFE = (
    "Previous answer violated the single read-only SQLite statement constraint. "
    "Return exactly one SELECT/CTE."
)
RETRY_FEEDBACK_PARSE = (
    "The previous answer did not contain a parseable <sql> block. "
    "Return exactly one SQL statement inside <sql>...</sql>."
)
RETRY_FEEDBACK_WRONG = (
    "The previous query executed successfully but did not match the expected result. "
    "Re-check the requested columns, Evidence, joins, filters, grouping and aggregation."
)


class DistillationFingerprintMismatchError(RuntimeError):
    pass


def _sanitize_sqlite_error(message: str) -> str:
    """Return a short SQLite error category/message without leaking gold data.

    This is a diagnostic aid for retry prompts. It never includes gold SQL or
    gold result rows.
    """
    lowered = (message or "").lower()
    categories = [
        "no such column",
        "no such table",
        "no such function",
        "ambiguous column",
        "no such database",
        "syntax error",
        "misuse",
        "table has no column",
    ]
    for category in categories:
        if category in lowered:
            return category
    if "near" in lowered and "syntax error" in lowered:
        return "syntax error"
    return (message or "").strip()[:160]


def _seed_for(global_seed: int, example_id: str, attempt_index: int) -> int:
    digest = hashlib.sha256(f"{global_seed}:{example_id}:{attempt_index}".encode()).hexdigest()
    return int(digest[:16], 16)


def _record_failure_reason(record: dict) -> str:
    if not record.get("parse_valid") or not record.get("candidate_sql"):
        return "parse_failed"
    if not record.get("safe"):
        return "unsafe_or_multiple_statements"
    if not record.get("execution_success"):
        return "execution_failed"
    if not record.get("execution_equivalent"):
        return "wrong_result"
    return "verified"


def _retry_feedback(record: dict) -> str:
    reason = _record_failure_reason(record)
    if reason in {"unsafe_or_multiple_statements"}:
        return RETRY_FEEDBACK_UNSAFE
    if reason == "execution_failed":
        category = record.get("execution_error_message_sanitized") or record.get(
            "execution_error_type"
        )
        if category:
            return f"Previous query failed with SQLite error category: {category}."
        return RETRY_FEEDBACK_PARSE
    if reason == "parse_failed":
        return RETRY_FEEDBACK_PARSE
    return RETRY_FEEDBACK_WRONG


@dataclass
class DistillationConfig:
    examples_path: Path
    registry_path: Path
    output_path: Path
    progress_path: Path | None = None
    teacher_model: str = "mock-teacher-1.0"
    teacher_model_revision: str = "unknown"
    teacher_prompt_version: str = "v1"
    generation_config: dict = field(default_factory=dict)
    num_candidates: int = 1
    max_samples: int | None = None
    resume: bool = False
    dry_run: bool = False
    require_plan: bool = False
    run_id: str = ""
    backend_name: str = "mock"
    backend_kwargs: dict = field(default_factory=dict)
    allowed_splits: frozenset[str] = frozenset({"train"})
    max_attempts: int = 1
    seed: int = 3407
    target_verified_examples: int | None = None


def compute_run_fingerprint(
    examples_path: Path,
    teacher_model: str,
    teacher_revision: str,
    prompt_version: str,
    generation_config: dict,
    num_candidates: int,
    max_attempts: int = 1,
    target_verified_examples: int | None = None,
) -> dict:
    return {
        "dataset_sha256": sha256_file(examples_path),
        "teacher_model": teacher_model,
        "teacher_model_revision": teacher_revision,
        "teacher_prompt_version": prompt_version,
        "generation_config": generation_config,
        "num_candidates": num_candidates,
        "max_attempts": max_attempts,
        "target_verified_examples": target_verified_examples,
    }


class DistillationPipeline:
    def __init__(
        self,
        config: DistillationConfig,
        environment: SQLExecutionEnvironment | None = None,
        backend: TeacherBackend | None = None,
    ):
        self.config = config
        self.run_id = config.run_id or uuid.uuid4().hex
        self.environment = environment or SQLExecutionEnvironment.from_registry(
            config.registry_path
        )
        self.backend = backend or self._default_backend()
        self.verifier = ResultEquivalenceVerifier()
        self.output_path = Path(config.output_path)
        self.progress_path = Path(
            config.progress_path or (self.output_path.with_suffix(".progress.json"))
        )
        self.manifest_path = self.output_path.with_suffix(".manifest.json")
        self.fingerprint = compute_run_fingerprint(
            examples_path=Path(config.examples_path),
            teacher_model=config.teacher_model,
            teacher_revision=config.teacher_model_revision,
            prompt_version=config.teacher_prompt_version,
            generation_config=config.generation_config,
            num_candidates=config.num_candidates,
            max_attempts=config.max_attempts,
            target_verified_examples=config.target_verified_examples,
        )

    def _default_backend(self) -> TeacherBackend:
        if self.config.backend_name == "mock":
            examples = load_examples(self.config.examples_path)
            oracle = {e.example_id: e.gold_sql for e in examples}
            return MockTeacherBackend(gold_oracle=oracle, **self.config.backend_kwargs)
        if self.config.backend_name == "transformers":
            return TransformersTeacherBackend(
                model_id=self.config.backend_kwargs.get("model_id", "Qwen/Qwen3-4B"),
                **{
                    key: value
                    for key, value in self.config.backend_kwargs.items()
                    if key != "model_id"
                },
            )
        raise ValueError(f"unknown backend {self.config.backend_name!r}")

    def _safe_context(self, example) -> dict:
        return {
            "example_id": example.example_id,
            "db_id": example.db_id,
            "split": example.split,
        }

    def _prompt(self, example, retry_feedback: str | None = None) -> str:
        prompt = build_prompt(
            question=example.question,
            schema_text=example.schema_text,
            db_id=example.db_id,
            include_plan=self.config.require_plan,
            evidence=example.evidence,
            prompt_version=self.config.teacher_prompt_version,
        )
        if retry_feedback:
            prompt = prompt + "\n\n" + retry_feedback
        return prompt

    def _completed_keys(self, records: list[dict]) -> set[tuple[str, int]]:
        return {
            (record.get("example_id"), int(record.get("candidate_index", -1)))
            for record in records
            if record.get("teacher_prompt_version") == self.config.teacher_prompt_version
        }

    def _check_resume_fingerprint(self) -> None:
        if not self.manifest_path.exists():
            raise DistillationFingerprintMismatchError(
                f"{self.manifest_path} is missing; refusing --resume because the run "
                "cannot be proven to come from the same dataset/teacher/generation "
                "fingerprint. Use a new output directory."
            )
        manifest = load_json(self.manifest_path)
        self.run_id = manifest.get("run_id", self.run_id)
        stored = manifest.get("run_fingerprint", {})
        if stored != self.fingerprint:
            raise DistillationFingerprintMismatchError(
                "run fingerprint mismatch; refusing --resume. "
                f"stored={stored} current={self.fingerprint}"
            )

    def run(self) -> dict:
        assert_training_splits(self.config.allowed_splits, policy_name="distillation")
        policy = SplitPolicy(
            allowed_splits=set(self.config.allowed_splits), policy_name="train_only"
        )
        all_examples = load_examples(self.config.examples_path)
        examples, split_report = policy.apply(all_examples, source_path=self.config.examples_path)
        if self.config.max_samples is not None:
            examples = examples[: max(0, int(self.config.max_samples))]

        existing_records: list[dict] = []
        if self.output_path.exists():
            existing_records = load_jsonl(self.output_path)
            if not self.config.resume:
                raise FileExistsError(
                    f"{self.output_path} already exists; pass --resume to continue, "
                    "or use a new output directory"
                )
            self._check_resume_fingerprint()
        completed = self._completed_keys(existing_records)
        adaptive = self.config.max_attempts > 1
        if adaptive:
            candidates_planned = len(examples) * self.config.max_attempts
        else:
            candidates_planned = len(examples) * self.config.num_candidates
        progress = {
            "examples_planned": len(examples),
            "examples_completed": 0,
            "candidates_planned": candidates_planned,
            "candidates_completed": len(completed),
        }
        if self.progress_path.exists():
            with contextlib.suppress(Exception):  # corrupt progress is rebuilt below
                progress = load_json(self.progress_path)

        if self.config.dry_run:
            if adaptive:
                to_generate = sum(
                    1
                    for example in examples
                    for attempt in range(self.config.max_attempts)
                    if (example.example_id, attempt) not in completed
                )
            else:
                to_generate = sum(
                    1
                    for example in examples
                    for index in range(self.config.num_candidates)
                    if (example.example_id, index) not in completed
                )
            return {
                "dry_run": True,
                "run_id": self.run_id,
                "examples_seen": len(examples),
                "already_completed": len(completed),
                "to_generate": to_generate,
                "output_path": str(self.output_path),
                "backend": self.backend.name,
                "split_report": split_report.as_dict(),
                "run_fingerprint": self.fingerprint,
            }

        if not self.manifest_path.exists():
            atomic_write_json(
                self.manifest_path,
                {
                    "status": "RUNNING",
                    "run_id": self.run_id,
                    "run_fingerprint": self.fingerprint,
                    "split_report": split_report.as_dict(),
                    "output_path": str(self.output_path),
                    "teacher_model": self.config.teacher_model,
                    "teacher_model_revision": self.config.teacher_model_revision,
                    "teacher_prompt_version": self.config.teacher_prompt_version,
                    "generation_config": self.config.generation_config,
                    "num_candidates": self.config.num_candidates,
                    "max_attempts": self.config.max_attempts,
                    "seed": self.config.seed,
                    "created_at": utc_now_iso(),
                },
            )

        generated = 0
        skipped = 0
        examples_completed = 0
        gold_cache: dict[str, object] = {}
        verified_example_ids = {
            r["example_id"] for r in existing_records if r.get("execution_equivalent")
        }
        for example in examples:
            example_done = False
            if adaptive:
                for attempt_index in range(self.config.max_attempts):
                    key = (example.example_id, attempt_index)
                    if key in completed:
                        skipped += 1
                        continue

                    seed = _seed_for(self.config.seed, example.example_id, attempt_index)
                    try:
                        import torch

                        torch.manual_seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(seed)
                    except Exception:  # noqa: BLE001 - seed is best effort in CPU-only tests
                        pass

                    retry_feedback = None
                    if attempt_index > 0:
                        previous = self._find_previous_record(example.example_id, attempt_index - 1)
                        if previous is not None:
                            retry_feedback = _retry_feedback(previous)

                    prompt = self._prompt(example, retry_feedback=retry_feedback)
                    candidates = self.backend.generate(
                        prompt,
                        context=self._safe_context(example),
                        num_candidates=1,
                    )
                    candidate_text = candidates[0]
                    record = self._verify_and_record(
                        example,
                        index=attempt_index,
                        candidate_text=candidate_text,
                        gold_cache=gold_cache,
                        attempt_index=attempt_index,
                        generation_seed=seed,
                    )
                    append_jsonl(self.output_path, [record])
                    completed.add(key)
                    generated += 1
                    if record["execution_equivalent"]:
                        verified_example_ids.add(example.example_id)
                        example_done = True
                        break
                if example_done or any(
                    (example.example_id, i) in completed for i in range(self.config.max_attempts)
                ):
                    examples_completed += 1
            else:
                all_indices = {
                    (example.example_id, index) for index in range(self.config.num_candidates)
                }
                if all_indices <= completed:
                    skipped += self.config.num_candidates
                    example_done = True
                else:
                    prompt = self._prompt(example)
                    candidates = self.backend.generate(
                        prompt,
                        context=self._safe_context(example),
                        num_candidates=self.config.num_candidates,
                    )
                    for index, candidate_text in enumerate(candidates):
                        key = (example.example_id, index)
                        if key in completed:
                            skipped += 1
                            continue
                        seed = _seed_for(self.config.seed, example.example_id, index)
                        record = self._verify_and_record(
                            example,
                            index=index,
                            candidate_text=candidate_text,
                            gold_cache=gold_cache,
                            attempt_index=0,
                            generation_seed=seed,
                        )
                        append_jsonl(self.output_path, [record])
                        completed.add(key)
                        generated += 1
                        if record["execution_equivalent"]:
                            verified_example_ids.add(example.example_id)
                            example_done = True
                            break
                examples_completed += 1

            progress = {
                "examples_planned": len(examples),
                "examples_completed": examples_completed,
                "candidates_planned": candidates_planned,
                "candidates_completed": len(completed),
                "last_example_id": example.example_id,
                "updated_at": utc_now_iso(),
            }
            atomic_write_json(self.progress_path, progress)
            if (
                self.config.target_verified_examples is not None
                and len(verified_example_ids) >= self.config.target_verified_examples
            ):
                break

        manifest = load_json(self.manifest_path) if self.manifest_path.exists() else {}
        manifest.update(
            {
                "status": "COMPLETED",
                "completed_at": utc_now_iso(),
                "run_id": self.run_id,
                "run_fingerprint": self.fingerprint,
                "split_report": split_report.as_dict(),
                "output_path": str(self.output_path),
                "teacher_model": self.config.teacher_model,
                "teacher_model_revision": self.config.teacher_model_revision,
                "teacher_prompt_version": self.config.teacher_prompt_version,
                "generation_config": self.config.generation_config,
                "num_candidates": self.config.num_candidates,
                "max_attempts": self.config.max_attempts,
                "seed": self.config.seed,
            }
        )
        atomic_write_json(self.manifest_path, manifest)
        self.backend.unload()
        return {
            "dry_run": False,
            "run_id": self.run_id,
            "examples_seen": len(examples),
            "generated": generated,
            "skipped": skipped,
            "output_path": str(self.output_path),
            "progress_path": str(self.progress_path),
            "manifest_path": str(self.manifest_path),
            "backend": self.backend.name,
            "split_report": split_report.as_dict(),
            "run_fingerprint": self.fingerprint,
        }

    def _find_previous_record(self, example_id: str, attempt_index: int) -> dict | None:
        if not self.output_path.exists():
            return None
        for record in load_jsonl(self.output_path):
            if (
                record.get("example_id") == example_id
                and int(record.get("candidate_index", -1)) == attempt_index
            ):
                return record
        return None

    def _verify_and_record(
        self,
        example,
        index: int,
        candidate_text: str,
        gold_cache: dict | None = None,
        attempt_index: int = 0,
        generation_seed: int | None = None,
    ) -> dict:
        parsed = parse_model_output(candidate_text, require_plan=self.config.require_plan)
        decision: SafetyDecision | None = validate_sql(parsed.sql) if parsed.sql else None
        safe = bool(decision and decision.safe)
        execution_success = False
        execution_equivalent = False
        verification: object | None = None
        execution_error_type = "none"
        execution_error_message_sanitized: str | None = None

        if safe and parsed.sql is not None:
            candidate_result = self.environment.execute(example.db_id, parsed.sql)
            if gold_cache is not None and example.example_id in gold_cache:
                gold_result = gold_cache[example.example_id]
            else:
                gold_result = self.environment.execute(example.db_id, example.gold_sql)
                if gold_cache is not None:
                    gold_cache[example.example_id] = gold_result
            execution_success = candidate_result.success
            if not execution_success:
                execution_error_type = candidate_result.error_type
                execution_error_message_sanitized = _sanitize_sqlite_error(
                    candidate_result.error_message
                )
            verification = self.verifier.verify(
                candidate=candidate_result,
                gold=gold_result,
                candidate_sql=parsed.sql,
                gold_sql=example.gold_sql,
                schema_tables=set(self.environment.table_names(example.db_id)),
            )
            execution_equivalent = verification.strict_equivalent

        sql_hash = None
        if parsed.sql:
            sql_hash = hashlib.sha256(parsed.sql.strip().casefold().encode()).hexdigest()

        record = DistillationRecord(
            example_id=example.example_id,
            teacher_model=self.config.teacher_model,
            teacher_model_revision=self.config.teacher_model_revision,
            teacher_prompt_version=self.config.teacher_prompt_version,
            candidate_index=index,
            raw_candidate_output=candidate_text,
            candidate_sql=parsed.sql,
            candidate_plan=parsed.plan,
            parse_valid=bool(parsed.sql) and not parsed.parse_error,
            safe=safe,
            execution_success=execution_success,
            execution_equivalent=execution_equivalent,
            generation_config=self.config.generation_config,
            created_at=utc_now_iso(),
            attempt_index=attempt_index,
            generation_seed=generation_seed,
            normalized_sql_hash=sql_hash,
            safety_error_type=decision.error_type
            if decision
            else ("format_error" if not parsed.sql else None),
            execution_error_type=execution_error_type,
            execution_error_message_sanitized=execution_error_message_sanitized,
            verification_kind=verification.kind if verification else None,
        )
        data = record.model_dump()
        data["retry_reason"] = _record_failure_reason(data)
        return DistillationRecord.model_validate(data).model_dump()
