"""Overnight phase 1.6 paired-data integrity tests.

These tests use the real signal-recovery artifacts and are skipped when the
artifacts are not present (e.g. a fresh clone before the experiment).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAIRED = ROOT / "artifacts/signal_recovery/paired/paired_manifest.json"
GOLD_TRAIN = ROOT / "artifacts/signal_recovery/gold_sft/dataset/train.jsonl"
DIST_TRAIN = ROOT / "artifacts/signal_recovery/distilled_sft/dataset/train.jsonl"
ALL_CAND = ROOT / "artifacts/signal_recovery/teacher_collection/all_candidates.jsonl"

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in [PAIRED, GOLD_TRAIN, DIST_TRAIN, ALL_CAND]),
    reason="signal recovery artifacts not present",
)


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_paired_manifest_has_88_ids():
    manifest = json.loads(PAIRED.read_text(encoding="utf-8"))
    assert manifest["paired_count"] == 88
    assert len(manifest["example_ids"]) == 88


def test_gold_distilled_row_counts_equal_paired_count():
    manifest = json.loads(PAIRED.read_text(encoding="utf-8"))
    gold_rows = _rows(GOLD_TRAIN)
    dist_rows = _rows(DIST_TRAIN)
    assert len(gold_rows) == len(manifest["example_ids"]) == 88
    assert len(dist_rows) == len(manifest["example_ids"]) == 88


def test_gold_distilled_serialized_inputs_equal():
    gold_rows = _rows(GOLD_TRAIN)
    dist_rows = _rows(DIST_TRAIN)
    assert [r["input"] for r in gold_rows] == [r["input"] for r in dist_rows]
    assert [r["instruction"] for r in gold_rows] == [r["instruction"] for r in dist_rows]


def test_all_candidates_verified_ids_match_paired_manifest():
    manifest = json.loads(PAIRED.read_text(encoding="utf-8"))
    paired_ids = set(manifest["example_ids"])
    verified_ids = {
        r["example_id"]
        for r in _rows(ALL_CAND)
        if r.get("execution_equivalent")
        and r.get("parse_valid")
        and r.get("safe")
        and r.get("execution_success")
        and r.get("candidate_sql")
    }
    assert verified_ids == paired_ids
