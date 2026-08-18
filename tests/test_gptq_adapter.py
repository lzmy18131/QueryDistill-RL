"""GPTQModel adapter tests (config, calibration contamination, status honesty)."""

from __future__ import annotations

import json

import pytest

from querydistill.data.dataset import assert_calibration_split
from querydistill.quantization.gptq import (
    GPTQConfig,
    GPTQRunner,
    check_compatibility,
)
from tests.helpers import sample_example


class _FakeTokenizer:
    def __call__(
        self, text, truncation=False, max_length=None, return_tensors=None, add_special_tokens=True
    ):
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}


def test_config_validation():
    assert GPTQConfig(base_model_path="x").validate() == []
    assert GPTQConfig(base_model_path="x", bits=5).validate()
    assert GPTQConfig(base_model_path="x", group_size=0).validate()


def test_calibration_dataset_selects_allowed_splits_only(tmp_path):
    path = tmp_path / "examples.jsonl"
    rows = [
        sample_example(example_id="train-1", split="train").model_dump(),
        sample_example(example_id="test-1", split="test", question="q2").model_dump(),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    from querydistill.quantization.gptq import build_calibration_dataset

    # Mixed file: dev/test rows are excluded; selected train rows are allowed.
    samples, report = build_calibration_dataset(path, _FakeTokenizer(), max_samples=2, max_length=8)
    assert samples == [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}]
    assert report["excluded_by_split"]["test"] == 1

    test_only = tmp_path / "test.jsonl"
    test_only.write_text(json.dumps(rows[1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no \\['calibration', 'train'\\] examples"):
        build_calibration_dataset(test_only, _FakeTokenizer(), max_samples=2)


def test_assert_calibration_split_direct():
    assert_calibration_split([sample_example(split="train")])
    assert_calibration_split([sample_example(split="calibration")])
    with pytest.raises(ValueError):
        assert_calibration_split([sample_example(split="dev")])


def test_runner_dry_run_writes_honest_status(tmp_path):
    config = GPTQConfig(
        base_model_path=str(tmp_path / "missing-base"),
        output_dir=str(tmp_path / "gptq"),
        dry_run=True,
    )
    result = GPTQRunner(config).run(dry_run=True)
    assert result["status"] == "DRY_RUN"
    assert result["quantized"] is False
    status = json.loads((tmp_path / "gptq" / "status.json").read_text(encoding="utf-8"))
    assert status["quantized"] is False


def test_runner_missing_base_is_blocked_not_faked(tmp_path):
    config = GPTQConfig(
        base_model_path=str(tmp_path / "missing-base"),
        output_dir=str(tmp_path / "gptq_blocked"),
        dry_run=False,
    )
    result = GPTQRunner(config).run(dry_run=False)
    assert result["status"] == "GPTQ_SMOKE_BLOCKED_NO_BASE_MODEL"
    assert result["quantized"] is False


def test_compatibility_report_missing_and_present(tmp_path):
    assert check_compatibility(tmp_path / "missing")["exists"] is False
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "quantization_config": {
                    "quant_method": "gptq",
                    "bits": 4,
                    "group_size": 128,
                    "desc_act": False,
                },
            }
        ),
        encoding="utf-8",
    )
    report = check_compatibility(model_dir)
    assert report["quantization_bits"] == 4
    assert report["quantization_backend"] == "gptq"
    assert "no PASS" in report["note"] or "serve time" in report["note"]
