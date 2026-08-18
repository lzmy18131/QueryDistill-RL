"""Evaluation metrics and harness tests (round-2 semantics)."""

from __future__ import annotations

import json

import pytest

from querydistill.data.split_policy import EvaluationSplitRequiredError
from querydistill.evaluation.errors import ErrorBucket
from querydistill.evaluation.harness import EvaluationHarness, MockModelBackend
from querydistill.evaluation.metrics import EvaluationMetrics, EvaluationRecord


def _records(**fields):
    payload = {
        "example_id": "e",
        "split": "test",
        "db_id": "tiny",
        "sql": "SELECT 1",
        "format_ok": True,
        "sql_parse_ok": True,
        "parse_ok": True,
        "safe": True,
        "execution_success": True,
        "execution_equivalent": False,
        "verification_partial": False,
        "verification_kind": "not_equivalent",
        "error_bucket": ErrorBucket.WRONG_RESULT.value,
        "latency_ms": 1.0,
        "exact_match": False,
    }
    payload.update(fields)
    return EvaluationRecord(**payload)


def test_aggregate_rates():
    records = [
        _records(example_id="a", execution_equivalent=True, error_bucket="none"),
        _records(example_id="b", execution_equivalent=False, error_bucket="wrong_result"),
        _records(
            example_id="c",
            execution_equivalent=False,
            execution_success=False,
            sql_parse_ok=False,
            parse_ok=False,
            format_ok=False,
            error_bucket="format_error",
        ),
        _records(
            example_id="d",
            execution_equivalent=False,
            execution_success=False,
            safe=False,
            sql_parse_ok=False,
            error_bucket="unsafe_sql",
        ),
    ]
    metrics = EvaluationMetrics(records).aggregate()
    assert metrics["execution_accuracy"] == 0.25
    assert metrics["valid_sql_rate"] == 0.5
    assert metrics["sql_parse_valid_rate"] == 0.5
    assert metrics["format_valid_rate"] == 0.75
    assert metrics["execution_success_rate"] == 0.5
    assert metrics["unsafe_sql_rate"] == 0.25
    assert metrics["error_buckets"]["wrong_result"] == 1
    assert metrics["error_buckets"]["unsafe_sql"] == 1
    assert metrics["mean_latency_ms"] == 1.0


def test_aggregate_empty_is_zero_not_nan():
    metrics = EvaluationMetrics([]).aggregate()
    assert metrics["execution_accuracy"] == 0.0
    assert metrics["mean_latency_ms"] is None


def test_per_split_accuracy():
    records = [
        _records(example_id="a", split="train", execution_equivalent=True, error_bucket="none"),
        _records(example_id="b", split="train"),
        _records(example_id="c", split="test", execution_equivalent=True, error_bucket="none"),
    ]
    per_split = EvaluationMetrics(records).aggregate()["per_split"]
    assert per_split["train"]["execution_accuracy"] == 0.5
    assert per_split["test"]["execution_accuracy"] == 1.0


def _harness(tiny_db, examples, strategy="gold", oracle=None):
    from querydistill.sql.environment import SQLExecutionEnvironment

    environment = SQLExecutionEnvironment({"tiny": tiny_db}, max_rows=100, max_execution_ms=1000)
    oracle = oracle or {e.example_id: e.gold_sql for e in examples}
    backend = MockModelBackend(strategy=strategy, gold_oracle=oracle)
    return EvaluationHarness(environment, backend, require_plan=False)


def _test_example(example):
    return example.model_copy(update={"split": "test"})


def test_mock_gold_backend_scores_100_on_tiny_fixture(tiny_db, tiny_example):
    examples = [
        _test_example(tiny_example),
        _test_example(
            tiny_example.model_copy(
                update={
                    "example_id": "ex-002",
                    "question": "Second question",
                    "gold_sql": "SELECT age FROM users ORDER BY id",
                }
            )
        ),
    ]
    metrics = _harness(tiny_db, examples).run(examples, split="test")
    aggregate = metrics.aggregate()
    assert aggregate["execution_accuracy"] == 1.0
    assert aggregate["exact_match_secondary"] == 1.0
    assert aggregate["unsafe_sql_rate"] == 0.0


def test_mock_wrong_backend_has_zero_accuracy(tiny_db, tiny_example):
    examples = [_test_example(tiny_example)]
    metrics = _harness(tiny_db, examples, strategy="wrong").run(examples, split="test")
    aggregate = metrics.aggregate()
    assert aggregate["execution_accuracy"] == 0.0
    assert aggregate["error_buckets"]["wrong_result"] == 1


def test_mock_unsafe_and_malformed_buckets(tiny_db, tiny_example):
    examples = [_test_example(tiny_example)]
    for strategy, bucket in (
        ("unsafe", ErrorBucket.UNSAFE_SQL.value),
        ("malformed", ErrorBucket.FORMAT_ERROR.value),
    ):
        metrics = _harness(tiny_db, examples, strategy=strategy).run(examples, split="test")
        aggregate = metrics.aggregate()
        assert aggregate["error_buckets"].get(bucket, 0) == 1


def test_exact_match_is_secondary_and_case_insensitive(tiny_db, tiny_example):
    examples = [tiny_example]
    record = _harness(tiny_db, examples).evaluate_one(tiny_example)
    assert record.exact_match is True
    assert record.execution_equivalent is True
    assert record.verification_kind in {"unordered_rows", "ordered_rows"}


def test_record_serializes():
    payload = _records().as_dict()
    assert json.dumps(payload)  # no serialization error
    assert payload["error_bucket"] == "wrong_result"


def test_evaluation_requires_explicit_split(tiny_db, tiny_example):
    with pytest.raises(EvaluationSplitRequiredError):
        _harness(tiny_db, [tiny_example]).run([tiny_example], split=None)


def test_eval_test_only_and_dev_only(tiny_db, tiny_example):
    train_like = tiny_example.model_copy(
        update={"example_id": "train-1", "split": "train", "question": "train q"}
    )
    test_like = tiny_example.model_copy(
        update={"example_id": "test-1", "split": "test", "question": "test q"}
    )
    harness = _harness(tiny_db, [train_like, test_like])
    metrics = harness.run([train_like, test_like], split="test")
    assert [r.example_id for r in metrics.records] == ["test-1"]
    with pytest.raises(EvaluationSplitRequiredError):
        harness.run([train_like, test_like], split="train")


def test_partial_equivalence_not_accuracy(tiny_db, tiny_example):
    # An empty structural partial must not count as execution accuracy.
    gold_empty = tiny_example.model_copy(
        update={
            "example_id": "empty-1",
            "split": "test",
            "gold_sql": "SELECT name FROM users WHERE age > 100",
            "question": "empty q",
        }
    )
    harness = _harness(
        tiny_db,
        [gold_empty],
        strategy="gold",
        oracle={"empty-1": "SELECT name FROM users WHERE age > 100"},
    )
    # Candidate (from oracle) returns empty too -> partial, not strict.
    record = harness.evaluate_one(gold_empty)
    assert record.verification_partial is True
    assert record.execution_equivalent is False
    metrics = EvaluationMetrics([record]).aggregate()
    assert metrics["execution_accuracy"] == 0.0
    assert metrics["partial_equivalence_rate"] == 1.0


def test_parse_valid_metric_requires_real_parse(tiny_db, tiny_example):
    malformed = _test_example(
        tiny_example.model_copy(update={"example_id": "bad", "question": "bad q"})
    )
    harness = _harness(tiny_db, [malformed], strategy="malformed")
    record = harness.evaluate_one(malformed)
    assert record.format_ok is False
    assert record.sql_parse_ok is False
    metrics = EvaluationMetrics([record]).aggregate()
    assert metrics["format_valid_rate"] == 0.0
    assert metrics["sql_parse_valid_rate"] == 0.0
    assert metrics["valid_sql_rate"] == 0.0
