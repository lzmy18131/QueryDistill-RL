"""Paired Gold-SFT / Distilled-SFT target construction (P0-3).

Distilled mode never falls back to gold SQL. Missing verified teacher targets
raise by default (strict). Multiple verified candidates for one example are
resolved by a deterministic selection policy (smallest candidate_index), never
by dict overwrite order. Both arms are built from exactly the same example ids
and a paired manifest is written for auditability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..utils import atomic_write_json, sha256_file, utc_now_iso
from .schema import DistillationRecord, Example


class DistilledTargetMissingError(KeyError):
    pass


@dataclass
class PairedTargets:
    example_ids: list[str]
    gold_targets: dict[str, str]
    distilled_targets: dict[str, str]
    requested_train_example_ids: list[str] = field(default_factory=list)
    dropped_missing_teacher: list[str] = field(default_factory=list)
    selection_policy: str = "min_candidate_index"
    dataset_sha256: str = ""
    requested_count: int = 0
    paired_count: int = 0

    @property
    def verified_teacher_coverage(self) -> float:
        if self.requested_count <= 0:
            return 0.0
        return self.paired_count / self.requested_count

    def write_manifest(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            path,
            {
                "requested_train_example_ids": self.requested_train_example_ids,
                "example_ids": self.example_ids,
                "requested_count": self.requested_count,
                "paired_count": self.paired_count,
                "dataset_sha256": self.dataset_sha256,
                "verified_teacher_coverage": self.verified_teacher_coverage,
                "dropped_missing_teacher": self.dropped_missing_teacher,
                "selection_policy": self.selection_policy,
                "created_at": utc_now_iso(),
            },
        )
        return path


def select_verified_candidates(
    records: list[DistillationRecord], policy: str = "min_candidate_index"
) -> dict[str, str]:
    """Deterministically select one verified parsed-SQL target per example."""
    if policy != "min_candidate_index":
        raise ValueError(f"unsupported selection policy {policy!r}")
    grouped: dict[str, list[DistillationRecord]] = {}
    for record in records:
        if (
            record.parse_valid
            and record.safe
            and record.execution_success
            and record.execution_equivalent
            and record.candidate_sql
        ):
            grouped.setdefault(record.example_id, []).append(record)
    targets: dict[str, str] = {}
    for example_id, candidates in grouped.items():
        candidates.sort(key=lambda record: (record.candidate_index, record.created_at))
        targets[example_id] = candidates[0].candidate_sql
    return targets


def build_paired_targets(
    train_examples: list[Example],
    records: list[DistillationRecord],
    examples_path: str | Path,
    require_all: bool = True,
    selection_policy: str = "min_candidate_index",
) -> PairedTargets:
    distilled = select_verified_candidates(records, policy=selection_policy)
    ids = [example.example_id for example in train_examples]
    missing = [example_id for example_id in ids if example_id not in distilled]

    if require_all and missing:
        raise DistilledTargetMissingError(
            "distilled mode is strict: no verified teacher candidate for examples "
            f"{missing[:10]}; refusing to fall back to gold SQL"
        )

    paired_ids = [example_id for example_id in ids if example_id in distilled]
    gold = {example.example_id: example.gold_sql for example in train_examples}
    return PairedTargets(
        example_ids=paired_ids,
        gold_targets={example_id: gold[example_id] for example_id in paired_ids},
        distilled_targets={example_id: distilled[example_id] for example_id in paired_ids},
        requested_train_example_ids=ids,
        dropped_missing_teacher=missing,
        selection_policy=selection_policy,
        dataset_sha256=sha256_file(Path(examples_path)),
        requested_count=len(ids),
        paired_count=len(paired_ids),
    )
