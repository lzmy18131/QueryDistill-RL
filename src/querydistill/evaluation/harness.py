"""Evaluation harness with swappable model backends.

The same harness evaluates Base / Gold-SFT / Distilled-SFT / GRPO / GPTQ so the
numbers are comparable by construction.

Round-2 hardening:
* evaluation requires an explicit split (dev or test) - never mixed/everything
* real model backends receive RealModelContext only (no gold SQL/results)
* only STRICT verifier equivalence counts as execution accuracy
"""

from __future__ import annotations

import re
import time
from typing import Any, Protocol, runtime_checkable

from ..data.schema import Example
from ..data.split_policy import require_explicit_eval_split
from ..outputs.context import RealModelContext
from ..outputs.parser import parse_model_output
from ..outputs.prompting import build_prompt
from ..sql.environment import SQLExecutionEnvironment
from ..sql.safety import validate_sql
from ..sql.verifier import ResultEquivalenceVerifier
from .errors import classify_error
from .metrics import EvaluationMetrics, EvaluationRecord


def normalize_sql_for_em(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip().casefold()


@runtime_checkable
class ModelBackend(Protocol):
    name: str

    def generate(self, prompt: str, context: dict[str, Any] | None = None) -> str: ...


class MockModelBackend:
    """Test-only rule-based backend.

    ``strategy="gold"`` uses a standalone gold oracle map; gold never enters the
    backend context. Other strategies emit deterministic wrong/unsafe/malformed/
    timeout completions.
    """

    name = "mock"

    def __init__(self, strategy: str = "gold", gold_oracle: dict[str, str] | None = None):
        if strategy not in {"gold", "wrong", "unsafe", "malformed", "timeout"}:
            raise ValueError(f"unknown mock strategy {strategy!r}")
        self.strategy = strategy
        self.gold_oracle = gold_oracle or {}

    def generate(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        context = context or {}
        if self.strategy == "gold":
            example_id = str(context.get("example_id", ""))
            sql = self.gold_oracle.get(example_id, "")
            if not sql:
                raise ValueError("mock 'gold' strategy requires gold_oracle[example_id]")
        elif self.strategy == "wrong":
            sql = "SELECT 1 AS wrong"
        elif self.strategy == "unsafe":
            sql = "DROP TABLE students"
        elif self.strategy == "timeout":
            sql = (
                "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
                "SELECT sum(x) FROM c"
            )
        else:
            return "I could not parse this question."
        return f"<sql>\n{sql}\n</sql>"


class TransformersModelBackend:
    name = "transformers"

    def __init__(self, model_path: str, max_new_tokens: int = 192):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.float16, device_map="auto"
        )
        self._model.eval()

    def generate(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        if context is not None:
            unsafe = set(context) - RealModelContext.safe_keys()
            if unsafe:
                raise ValueError(
                    f"evaluation backend received unsafe context keys: {sorted(unsafe)}"
                )
        self.load()
        inputs = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._model.device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        return self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None


class EvaluationHarness:
    def __init__(
        self,
        environment: SQLExecutionEnvironment,
        backend: ModelBackend,
        require_plan: bool = False,
    ):
        self.environment = environment
        self.backend = backend
        self.require_plan = require_plan
        self.verifier = ResultEquivalenceVerifier()

    def _safe_context(self, example: Example) -> dict:
        return RealModelContext(
            example_id=example.example_id, db_id=example.db_id, split=example.split
        ).as_dict()

    def evaluate_one(self, example: Example) -> EvaluationRecord:
        prompt = build_prompt(
            question=example.question,
            schema_text=example.schema_text,
            db_id=example.db_id,
            include_plan=self.require_plan,
        )
        started = time.monotonic()
        output = self.backend.generate(prompt, context=self._safe_context(example))
        latency_ms = (time.monotonic() - started) * 1000.0

        parsed = parse_model_output(output, require_plan=self.require_plan)
        decision = validate_sql(parsed.sql) if parsed.sql else None
        execution = None
        verification = None
        safe = bool(decision and decision.safe)
        gold = None

        if safe and parsed.sql is not None:
            execution = self.environment.execute(example.db_id, parsed.sql)
            gold = self.environment.execute(example.db_id, example.gold_sql)
            verification = self.verifier.verify(
                candidate=execution,
                gold=gold,
                candidate_sql=parsed.sql,
                gold_sql=example.gold_sql,
                schema_tables=set(self.environment.table_names(example.db_id)),
            )

        bucket = classify_error(parsed, decision, execution, verification)
        exact_match = bool(parsed.sql) and normalize_sql_for_em(parsed.sql) == normalize_sql_for_em(
            example.gold_sql
        )
        return EvaluationRecord(
            example_id=example.example_id,
            split=example.split,
            db_id=example.db_id,
            sql=parsed.sql,
            format_ok=parsed.format_ok,
            sql_parse_ok=bool(
                decision and decision.error_type not in {"syntax_error", "format_error"}
            )
            if decision
            else False,
            parse_ok=bool(parsed.sql),
            safe=safe,
            execution_success=bool(execution and execution.success),
            execution_equivalent=bool(verification and verification.strict_equivalent),
            verification_partial=bool(verification and verification.partial_credit),
            verification_kind=verification.kind if verification else "none",
            error_bucket=bucket.value,
            latency_ms=round(latency_ms, 2),
            exact_match=exact_match,
            safety_error_type=decision.error_type if decision else "none",
            execution_error_type=execution.error_type if execution else "none",
            row_count=execution.row_count if execution else 0,
            gold_row_count=gold.row_count if gold else 0,
        )

    def run(
        self,
        examples: list[Example],
        split: str | None = None,
        max_examples: int | None = None,
    ) -> EvaluationMetrics:
        split = require_explicit_eval_split(split)
        selected = [example for example in examples if example.split == split]
        if max_examples is not None:
            selected = selected[: max(0, int(max_examples))]
        records = [self.evaluate_one(example) for example in selected]
        return EvaluationMetrics(records=records)
