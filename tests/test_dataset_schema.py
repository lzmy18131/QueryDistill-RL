"""Dataset schema, adapters, and calibration contamination guard tests."""

from __future__ import annotations

import pytest

from querydistill.data.dataset import (
    assert_calibration_split,
    build_prompt_rows,
    build_sft_rows,
    to_hf_dataset,
    tokenize_calibration,
)
from querydistill.data.schema import (
    DistillationRecord,
    DuplicateExampleError,
    load_distillation_records,
    load_examples,
)
from tests.helpers import sample_example


def test_example_schema_required_fields():
    example = sample_example()
    payload = example.model_dump()
    for field in (
        "example_id",
        "db_id",
        "question",
        "schema_text",
        "gold_sql",
        "split",
        "source",
        "source_version",
    ):
        assert field in payload


def test_example_schema_rejects_bad_split_and_db_id():
    with pytest.raises(ValueError):
        sample_example(split="validation")
    with pytest.raises(ValueError):
        sample_example(db_id="../../evil")
    with pytest.raises(ValueError):
        sample_example(question="")


def test_distillation_record_schema_fields():
    record = DistillationRecord(
        example_id="ex-1",
        teacher_model="mock",
        teacher_model_revision="r1",
        teacher_prompt_version="v1",
        candidate_index=0,
        raw_candidate_output="<sql>SELECT 1</sql>",
        candidate_sql="SELECT 1",
        candidate_plan=None,
        parse_valid=True,
        safe=True,
        execution_success=True,
        execution_equivalent=True,
        generation_config={"temperature": 0.7},
        created_at="2026-08-17T00:00:00+00:00",
    )
    payload = record.model_dump()
    for field in (
        "example_id",
        "teacher_model",
        "teacher_model_revision",
        "teacher_prompt_version",
        "candidate_index",
        "raw_candidate_output",
        "candidate_sql",
        "candidate_plan",
        "parse_valid",
        "safe",
        "execution_success",
        "execution_equivalent",
        "generation_config",
        "created_at",
    ):
        assert field in payload
    assert payload["candidate_sql"] == "SELECT 1"


def test_load_examples_detects_duplicates(tmp_path):
    path = tmp_path / "examples.jsonl"
    record = sample_example().model_dump()
    path.write_text(
        "\n".join(__import__("json").dumps(record) for _ in range(2)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DuplicateExampleError):
        load_examples(path)


def test_load_examples_split_mismatch(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text(
        __import__("json").dumps(sample_example(split="test").model_dump()) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="split mismatch"):
        load_examples(path, declared_split="train")


def test_distillation_records_roundtrip(tmp_path):
    path = tmp_path / "distilled.jsonl"
    record = DistillationRecord(
        example_id="ex-1",
        teacher_model="mock",
        teacher_prompt_version="v1",
        candidate_index=0,
        raw_candidate_output="<sql>SELECT 1</sql>",
        candidate_sql="SELECT 1",
        parse_valid=True,
        safe=True,
        execution_success=True,
        execution_equivalent=False,
        created_at="t",
    )
    path.write_text(__import__("json").dumps(record.model_dump()) + "\n", encoding="utf-8")
    loaded = load_distillation_records(path)
    assert len(loaded) == 1
    assert loaded[0].candidate_sql == "SELECT 1"
    assert loaded[0].raw_candidate_output == "<sql>SELECT 1</sql>"


def test_to_hf_dataset_columns():
    dataset = to_hf_dataset([sample_example()])
    for column in ("example_id", "db_id", "question", "schema_text", "gold_sql", "split"):
        assert column in dataset.column_names


def test_build_prompt_rows_registry():
    examples = [sample_example(example_id=f"ex-{i}", split="train") for i in range(3)]
    rows, registry = build_prompt_rows(examples)
    assert len(rows) == 3
    for row in rows:
        example = registry[row["example_id"]]
        assert example.example_id == row["example_id"]
        assert example.gold_sql not in row["prompt"]  # policy prompt never contains gold


def test_build_sft_rows_target_switching():
    example = sample_example()
    gold_rows = build_sft_rows([example])
    assert example.gold_sql in gold_rows[0]["output"]
    distilled_rows = build_sft_rows([example], target_sql_by_id={"ex-001": "SELECT 2"})
    assert "SELECT 2" in distilled_rows[0]["output"]


class _FakeTokenizer:
    def __call__(
        self, text, truncation=False, max_length=None, return_tensors=None, add_special_tokens=True
    ):
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}


def test_tokenize_calibration_and_split_guard():
    tokenizer = _FakeTokenizer()
    train = [sample_example(example_id="a", split="train")]
    samples = tokenize_calibration(tokenizer, ["question text"], max_length=8)
    assert samples == [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}]
    assert_calibration_split(train)
    contaminated = train + [sample_example(example_id="b", split="test")]
    with pytest.raises(ValueError, match="calibration"):
        assert_calibration_split(contaminated)
