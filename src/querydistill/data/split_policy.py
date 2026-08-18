"""Fail-closed split policy shared by every training / distillation path.

Training paths are only allowed to see ``train`` (GPTQ calibration:
``train`` + ``calibration``). Dev/test examples are excluded and recorded by
default; a split that is not part of the project schema is rejected outright.

Evaluation is the opposite: it must explicitly name ``dev`` or ``test`` and
refuses to run without a split argument.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from .schema import ALLOWED_SPLITS, Example

TRAINING_ALLOWED = frozenset({"train"})
CALIBRATION_ALLOWED = frozenset({"train", "calibration"})
EVALUATION_ALLOWED = frozenset({"dev", "test", "validation_tuning"})
FORBIDDEN_TRAINING_SPLITS = frozenset({"validation_tuning"})


class UnknownSplitError(ValueError):
    """A split outside the project schema reached a fail-closed boundary."""


class EvaluationSplitRequiredError(ValueError):
    """Formal evaluation must explicitly select dev or test."""


class TrainingSplitViolation(ValueError):
    """A training/calibration boundary received a disallowed split."""


def assert_training_splits(allowed_splits, policy_name: str = "train_only") -> frozenset[str]:
    """Fail-closed invariant: training entry points may only see ``train``."""
    normalized = frozenset(allowed_splits)
    if normalized != TRAINING_ALLOWED:
        raise TrainingSplitViolation(
            f"{policy_name} requires allowed_splits == {{'train'}}, got {sorted(normalized)}"
        )
    return normalized


def assert_calibration_splits(allowed_splits, policy_name: str = "calibration") -> frozenset[str]:
    """Fail-closed invariant: calibration may only see train/calibration."""
    normalized = frozenset(allowed_splits)
    if not normalized <= CALIBRATION_ALLOWED:
        raise TrainingSplitViolation(
            f"{policy_name} allows only train/calibration, got {sorted(normalized)}"
        )
    return normalized


@dataclass
class SplitReport:
    policy: str
    source_path: str
    total: int = 0
    included: int = 0
    excluded_by_split: dict[str, int] = field(default_factory=dict)
    unknown_example_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SplitPolicy:
    """Select examples by split with explicit accounting.

    ``reject_disallowed=True`` (the fail-closed default) raises for any split
    not declared in the project schema. Known-but-disallowed splits
    (dev/test during training) are excluded and recorded, never silently used.
    """

    allowed_splits: frozenset[str] = TRAINING_ALLOWED
    policy_name: str = "train_only"

    def apply(
        self, examples: list[Example], source_path: str | Path = ""
    ) -> tuple[list[Example], SplitReport]:
        report = SplitReport(
            policy=self.policy_name, source_path=str(source_path), total=len(examples)
        )
        selected: list[Example] = []
        for example in examples:
            if example.split not in ALLOWED_SPLITS:
                report.unknown_example_ids.append(example.example_id)
                continue
            if self.allowed_splits == TRAINING_ALLOWED and example.split in FORBIDDEN_TRAINING_SPLITS:
                raise TrainingSplitViolation(
                    f"{self.policy_name} training path received forbidden split "
                    f"{example.split!r} for example {example.example_id!r}"
                )
            if example.split in self.allowed_splits:
                selected.append(example)
            else:
                report.excluded_by_split[example.split] = (
                    report.excluded_by_split.get(example.split, 0) + 1
                )
        if report.unknown_example_ids:
            raise UnknownSplitError(
                f"{self.policy_name} policy refused unknown splits for examples: "
                f"{report.unknown_example_ids[:10]}"
            )
        report.included = len(selected)
        return selected, report


def require_explicit_eval_split(split: str | None) -> str:
    """Evaluation must name dev or test explicitly."""
    if split not in EVALUATION_ALLOWED:
        raise EvaluationSplitRequiredError(
            f"evaluation requires an explicit split: one of {sorted(EVALUATION_ALLOWED)}; got {split!r}"
        )
    return split


def select_examples_for_split(examples: list[Example], split: str) -> list[Example]:
    split = require_explicit_eval_split(split)
    return [example for example in examples if example.split == split]
