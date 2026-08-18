"""Dataset adapters shared by SFT / GRPO / GPTQ calibration paths.

Round-2 hardening:

* GRPO rows carry example_id + db_id; identity mapping is example_id -> Example
  (raw prompt strings are never the key, duplicate prompts are safe).
* Distilled-SFT targets never fall back to gold SQL.
* The default protocol is ``<sql>...</sql>`` without a placeholder plan; when a
  plan is requested it is derived from the real SQL AST.
* Prompt token budgets are enforced by a deterministic schema-truncation policy.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import sqlglot
from sqlglot import exp

from ..outputs.prompting import apply_student_chat_template, build_prompt
from ..sql.environment import SQLExecutionEnvironment
from .leakage import LeakageGuard
from .paired import DistilledTargetMissingError, select_verified_candidates
from .schema import Example, load_distillation_records, load_examples

CALIBRATION_ALLOWED_SPLITS = frozenset({"train", "calibration"})


def examples_to_list(examples: list[Example]) -> list[dict]:
    return [example.model_dump() for example in examples]


def to_hf_dataset(examples: list[Example]):
    """Convert examples into a HuggingFace ``datasets.Dataset`` (lazy import)."""
    from datasets import Dataset

    return Dataset.from_list([example.model_dump() for example in examples])


def plan_from_sql(sql: str, dialect: str = "sqlite") -> str:
    """Deterministic compact plan derived from the real SQL AST.

    Never a placeholder and never hidden chain-of-thought.
    """
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:  # noqa: BLE001 - malformed SQL yields an empty plan
        return ""
    lines = ["tables: " + ", ".join(sorted({t.name for t in tree.find_all(exp.Table) if t.name}))]
    joins = sorted({str(join).strip()[:80] for join in tree.find_all(exp.Join)})
    if joins:
        lines.append("joins: " + " | ".join(joins))
    where = tree.find(exp.Where)
    if where is not None:
        lines.append("filters: " + str(where.this)[:120])
    group = tree.find(exp.Group)
    if group is not None:
        lines.append("grouping: " + str(group)[:80])
    order = tree.find(exp.Order)
    if order is not None:
        lines.append("ordering: " + str(order)[:80])
    return "\n".join(lines)


def _format_target(sql: str, include_plan: bool) -> str:
    if include_plan:
        plan = plan_from_sql(sql)
        return f"<plan>\n{plan}\n</plan>\n<sql>\n{sql}\n</sql>" if plan else f"<sql>\n{sql}\n</sql>"
    return f"<sql>\n{sql}\n</sql>"


def build_sft_rows(
    examples: list[Example],
    target_sql_by_id: dict[str, str] | None = None,
    include_plan: bool = False,
) -> list[dict]:
    """Alpaca-style rows for LLaMA-Factory: instruction/input/output.

    Gold-SFT passes ``target_sql_by_id=None``. Distilled-SFT must pass a complete
    verified target mapping; a missing example_id raises immediately - gold SQL
    is NEVER used as a fallback for distilled mode.
    """
    rows: list[dict] = []
    for example in examples:
        if target_sql_by_id is None:
            target_sql = example.gold_sql
        else:
            target_sql = target_sql_by_id.get(example.example_id)
            if not target_sql:
                raise DistilledTargetMissingError(
                    f"no verified distilled target for {example.example_id}; "
                    "refusing to fall back to gold SQL"
                )
        user_content = build_prompt(
            question=example.question,
            schema_text=example.schema_text,
            db_id=example.db_id,
            include_plan=include_plan,
            evidence=example.evidence,
        )
        rows.append(
            {
                "instruction": "",
                "input": user_content,
                "output": _format_target(target_sql, include_plan),
                "example_id": example.example_id,
                "db_id": example.db_id,
            }
        )
    return rows


def load_verified_distilled_targets(records_path: Path) -> dict[str, str]:
    """Deterministic verified parsed-SQL targets (min candidate_index policy)."""
    records = load_distillation_records(records_path)
    return select_verified_candidates(records, policy="min_candidate_index")


def build_prompt_rows(
    examples: list[Example],
    include_plan: bool = False,
    tokenizer=None,
    max_prompt_tokens: int | None = None,
) -> tuple[list[dict], dict[str, Example]]:
    """Rows for GRPO plus an example_id -> Example registry.

    Dataset rows carry ``example_id`` / ``db_id`` / ``prompt`` so the reward
    function can resolve identity from metadata even with duplicate prompts.
    """
    rows: list[dict] = []
    registry: dict[str, Example] = {}
    for example in examples:
        prompt = build_prompt(
            question=example.question,
            schema_text=example.schema_text,
            db_id=example.db_id,
            include_plan=include_plan,
            evidence=example.evidence,
        )
        truncated = False
        if tokenizer is not None:
            if max_prompt_tokens is not None:
                prompt, truncated = truncate_prompt_to_token_budget(
                    tokenizer=tokenizer,
                    question=example.question,
                    schema_text=example.schema_text,
                    db_id=example.db_id,
                    include_plan=include_plan,
                    evidence=example.evidence,
                    max_prompt_tokens=max_prompt_tokens,
                )
            prompt = apply_student_chat_template(
                tokenizer,
                question=example.question,
                schema_text=example.schema_text,
                db_id=example.db_id,
                include_plan=include_plan,
                evidence=example.evidence,
                add_generation_prompt=True,
                tokenize=False,
            )
        rows.append(
            {
                "example_id": example.example_id,
                "db_id": example.db_id,
                "prompt": prompt,
                "prompt_truncated": truncated,
            }
        )
        registry[example.example_id] = example
    return rows, registry


def truncate_prompt_to_token_budget(
    tokenizer,
    question: str,
    schema_text: str,
    db_id: str,
    include_plan: bool = False,
    max_prompt_tokens: int = 512,
    evidence: str = "",
) -> tuple[str, bool]:
    """Deterministic token-budget prompt truncation.

    Policy (documented): keep the question and the database header verbatim;
    if the schema exceeds the budget, keep whole schema lines in their original
    order from the top (earliest CREATE TABLE statements) and drop trailing
    lines. The returned flag records that truncation happened.
    """
    full = build_prompt(
        question=question,
        schema_text=schema_text,
        db_id=db_id,
        include_plan=include_plan,
        evidence=evidence,
    )
    token_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if len(token_ids) <= max_prompt_tokens:
        return full, False

    lines = schema_text.splitlines()
    low, high = 0, len(lines)
    while low < high:
        mid = (low + high + 1) // 2
        candidate = build_prompt(
            question=question,
            schema_text="\n".join(lines[:mid]),
            db_id=db_id,
            include_plan=include_plan,
            evidence=evidence,
        )
        if len(tokenizer(candidate, add_special_tokens=False)["input_ids"]) <= max_prompt_tokens:
            low = mid
        else:
            high = mid - 1
    truncated = build_prompt(
        question=question,
        schema_text="\n".join(lines[:low]) if low > 0 else "(schema truncated to fit budget)",
        db_id=db_id,
        include_plan=include_plan,
        evidence=evidence,
    )
    return truncated, True


def assert_calibration_split(
    examples: list[Example], allowed: frozenset[str] = CALIBRATION_ALLOWED_SPLITS
) -> None:
    """GPTQ calibration must never see dev/test examples (test contamination guard)."""
    bad = [example.example_id for example in examples if example.split not in allowed]
    if bad:
        raise ValueError(f"calibration dataset contains non-{sorted(allowed)} examples: {bad}")


def tokenize_calibration(
    tokenizer,
    texts: Iterable[str],
    max_length: int = 512,
) -> list[dict]:
    """Tokenize calibration texts into GPTQModel-compatible list of dicts."""
    samples: list[dict] = []
    for text in texts:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
            add_special_tokens=True,
        )
        samples.append(
            {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
        )
    return samples


def load_and_guard_examples(examples_path: Path) -> list[Example]:
    examples = load_examples(examples_path)
    guard = LeakageGuard()
    guard.assert_clean(examples)
    return examples


def environment_for_registry(registry_path: Path) -> SQLExecutionEnvironment:
    return SQLExecutionEnvironment.from_registry(registry_path)
