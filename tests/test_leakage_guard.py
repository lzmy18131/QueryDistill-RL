"""LeakageGuard P0 tests."""

from __future__ import annotations

import json

import pytest

from querydistill.data.leakage import (
    LeakageError,
    LeakageGuard,
    normalize_question,
    normalize_sql,
)
from tests.helpers import sample_example


def test_id_overlap_between_splits_is_detected():
    examples = [
        sample_example(example_id="same", split="train"),
        sample_example(example_id="same", split="test", question="different question?"),
    ]
    report = LeakageGuard().audit_examples(examples)
    assert not report.clean
    assert any(v["rule_id"] == "id_overlap" for v in report.violations)


def test_exact_and_normalized_question_overlap_detected():
    examples = [
        sample_example(example_id="a", split="train", question="How many users exist?"),
        sample_example(
            example_id="b",
            split="test",
            question="  how   many USERS exist?? ",
            gold_sql="SELECT 1",
        ),
    ]
    report = LeakageGuard().audit_examples(examples)
    ids = {v["rule_id"] for v in report.violations}
    assert "question_overlap_normalized" in ids


def test_prompt_leakage_gold_sql_detected():
    example = sample_example(gold_sql="SELECT name FROM users")
    prompt = "Schema:\nusers(...)\nHint: the answer is SELECT name FROM users\n"
    report = LeakageGuard().audit_examples([example], prompts_by_id={"ex-001": prompt})
    assert any(v["rule_id"] == "prompt_leakage_gold_sql" for v in report.violations)


def test_prompt_without_gold_is_clean():
    example = sample_example(gold_sql="SELECT name FROM users")
    prompt = "Schema:\nCREATE TABLE users (...)\nQuestion: list names"
    report = LeakageGuard().audit_examples([example], prompts_by_id={"ex-001": prompt})
    assert report.clean


def test_gold_result_leakage_detected():
    example = sample_example(gold_sql="SELECT name FROM users")
    prompt = "Use known answer 'Xylophone-42' in your reasoning"
    report = LeakageGuard().audit_examples(
        [example],
        prompts_by_id={"ex-001": prompt},
        gold_results_by_id={"ex-001": [["Xylophone-42"]]},
    )
    assert any(v["rule_id"] == "prompt_leakage_gold_result" for v in report.violations)


def test_file_split_mismatch_detected(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(sample_example(split="test").model_dump()) + "\n", encoding="utf-8")
    report = LeakageGuard().check_file_split_mismatch(path, declared_split="train")
    assert not report.clean


def test_distillation_from_test_split_detected():
    examples = [
        sample_example(example_id="train-1", split="train"),
        sample_example(example_id="test-1", split="test", question="q2"),
    ]
    report = LeakageGuard().audit_examples(examples, candidate_example_ids=["test-1"])
    assert any(v["rule_id"] == "distillation_from_test" for v in report.violations)


def test_assert_clean_raises_typed_error():
    examples = [
        sample_example(example_id="x", split="train"),
        sample_example(example_id="x", split="test", question="q2"),
    ]
    with pytest.raises(LeakageError):
        LeakageGuard().assert_clean(examples)


def test_clean_dataset_passes():
    examples = [
        sample_example(example_id="a", split="train", question="First question"),
        sample_example(
            example_id="b", split="test", question="Second question", gold_sql="SELECT 2"
        ),
    ]
    report = LeakageGuard().assert_clean(examples)
    assert report.clean
    assert "id_overlap" in report.checks


def test_normalization_helpers():
    assert normalize_question("  Hello,   WORLD! ") == "hello world"
    assert normalize_question("ＡＢＣ") == "abc"
    assert normalize_sql("SELECT   a\nFROM t;") == "select a from t;"


def test_casefolded_gold_sql_leak_detected():
    example = sample_example(gold_sql="SELECT Name FROM Users")
    prompt = "hint: select name from users"
    report = LeakageGuard().audit_examples([example], prompts_by_id={"ex-001": prompt})
    assert not report.clean
