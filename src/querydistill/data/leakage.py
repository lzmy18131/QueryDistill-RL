"""LeakageGuard (P0).

Enforced invariants:

* no example-id overlap between any two splits
* no exact or normalized question overlap between splits
* gold SQL never appears in a policy prompt (prompt leakage)
* gold execution result values never appear in a policy prompt
* no test-split record inside a train/dev file (split mismatch)
* no teacher candidate generated from a test-split example

Reward/evaluator code may see gold SQL and gold results; the model policy may
not. See ``docs/REWARD_DESIGN.md`` and ``tests/test_leakage_guard.py``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from .schema import Example, load_examples

_WHITESPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)


def normalize_question(question: str) -> str:
    """NFKC + casefold + punctuation removal + whitespace collapse."""
    text = unicodedata.normalize("NFKC", question).casefold()
    text = _NON_WORD_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_sql(sql: str) -> str:
    """Whitespace-collapsed, casefolded SQL for containment checks."""
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", sql)).strip().casefold()


@dataclass
class LeakageReport:
    violations: list[dict] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.violations

    def add(self, rule_id: str, severity: str, detail: str) -> None:
        self.violations.append({"rule_id": rule_id, "severity": severity, "detail": detail})

    def as_dict(self) -> dict:
        return {"clean": self.clean, "violations": self.violations, "checks": self.checks}


class LeakageError(RuntimeError):
    def __init__(self, report: LeakageReport):
        self.report = report
        details = "; ".join(v["detail"] for v in report.violations[:5])
        super().__init__(f"leakage guard violations ({len(report.violations)}): {details}")


class LeakageGuard:
    def check_id_overlap(
        self, examples: list[Example], report: LeakageReport | None = None
    ) -> LeakageReport:
        report = report or LeakageReport()
        report.checks.append("id_overlap")
        by_split: dict[str, set[str]] = {}
        for example in examples:
            by_split.setdefault(example.split, set()).add(example.example_id)
        for left, right in combinations(sorted(by_split), 2):
            overlap = sorted(by_split[left] & by_split[right])
            if overlap:
                report.add(
                    "id_overlap", "error", f"splits {left}/{right} share example ids: {overlap}"
                )
        return report

    def check_question_overlap(
        self, examples: list[Example], report: LeakageReport | None = None
    ) -> LeakageReport:
        report = report or LeakageReport()
        report.checks.append("question_overlap")
        by_split: dict[str, list[Example]] = {}
        for example in examples:
            by_split.setdefault(example.split, []).append(example)
        for left, right in combinations(sorted(by_split), 2):
            right_exact = {e.question for e in by_split[right]}
            right_normalized = {normalize_question(e.question) for e in by_split[right]}
            for example in by_split[left]:
                if example.question in right_exact:
                    report.add(
                        "question_overlap_exact",
                        "error",
                        f"exact question overlap between {left}/{right}: {example.question!r}",
                    )
                normalized = normalize_question(example.question)
                if normalized in right_normalized:
                    report.add(
                        "question_overlap_normalized",
                        "error",
                        f"normalized question overlap between {left}/{right}: {example.question!r}",
                    )
        return report

    def check_prompt_leakage(
        self, example: Example, prompt: str, report: LeakageReport | None = None
    ) -> LeakageReport:
        report = report or LeakageReport()
        report.checks.append("prompt_leakage_gold_sql")
        prompt_normalized = normalize_sql(prompt)
        gold_normalized = normalize_sql(example.gold_sql)
        if gold_normalized in prompt_normalized:
            report.add(
                "prompt_leakage_gold_sql",
                "error",
                f"gold SQL of {example.example_id} appears inside the policy prompt",
            )
        if example.gold_sql.strip() in prompt:
            report.add(
                "prompt_leakage_gold_sql_raw",
                "error",
                f"raw gold SQL of {example.example_id} appears inside the policy prompt",
            )
        return report

    def check_result_leakage(
        self,
        example: Example,
        prompt: str,
        gold_rows: list[list] | None,
        report: LeakageReport | None = None,
    ) -> LeakageReport:
        report = report or LeakageReport()
        report.checks.append("prompt_leakage_gold_result")
        if not gold_rows:
            return report
        for row in gold_rows:
            for value in row:
                if isinstance(value, str) and len(value) >= 6 and value in prompt:
                    report.add(
                        "prompt_leakage_gold_result",
                        "error",
                        f"gold result value {value!r} of {example.example_id} appears in prompt",
                    )
        return report

    def check_file_split_mismatch(self, path: str | Path, declared_split: str) -> LeakageReport:
        report = LeakageReport()
        report.checks.append("file_split_mismatch")
        try:
            load_examples(path, declared_split=declared_split)
        except ValueError as exc:
            report.add("file_split_mismatch", "error", str(exc))
        return report

    def check_distillation_from_test(
        self,
        examples: list[Example],
        candidate_example_ids: list[str],
        report: LeakageReport | None = None,
    ) -> LeakageReport:
        report = report or LeakageReport()
        report.checks.append("distillation_from_test")
        test_ids = {e.example_id for e in examples if e.split == "test"}
        leaked = sorted(test_ids & set(candidate_example_ids))
        if leaked:
            report.add(
                "distillation_from_test",
                "error",
                f"teacher candidates were generated from test-split examples: {leaked}",
            )
        return report

    def audit_examples(
        self,
        examples: list[Example],
        prompts_by_id: dict[str, str] | None = None,
        gold_results_by_id: dict[str, list[list]] | None = None,
        candidate_example_ids: list[str] | None = None,
    ) -> LeakageReport:
        report = LeakageReport()
        self.check_id_overlap(examples, report)
        self.check_question_overlap(examples, report)
        prompts_by_id = prompts_by_id or {}
        gold_results_by_id = gold_results_by_id or {}
        for example in examples:
            prompt = prompts_by_id.get(example.example_id)
            if prompt is not None:
                self.check_prompt_leakage(example, prompt, report)
                if example.example_id in gold_results_by_id:
                    self.check_result_leakage(
                        example, prompt, gold_results_by_id[example.example_id], report
                    )
        if candidate_example_ids is not None:
            self.check_distillation_from_test(examples, candidate_example_ids, report)
        return report

    def assert_clean(
        self,
        examples: list[Example],
        prompts_by_id: dict[str, str] | None = None,
        gold_results_by_id: dict[str, list[list]] | None = None,
        candidate_example_ids: list[str] | None = None,
    ) -> LeakageReport:
        report = self.audit_examples(
            examples, prompts_by_id, gold_results_by_id, candidate_example_ids
        )
        if not report.clean:
            raise LeakageError(report)
        return report
