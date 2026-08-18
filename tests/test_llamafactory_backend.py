"""LLaMA-Factory backend tests (YAML, dataset registration, target modules)."""

from __future__ import annotations

import json

import pytest
import yaml

from querydistill.training.llamafactory_backend import (
    QLoRAConfig,
    build_command,
    build_yaml,
    detect_target_modules,
    prepare_dataset_dir,
    run_llamafactory,
    validate_target_modules,
)


class _FakeConfig:
    architectures = ["Qwen3ForCausalLM"]


def test_resolved_dict_contains_qlora_fields():
    config = QLoRAConfig(max_steps=3)
    payload = config.resolved_dict()
    assert payload["stage"] == "sft"
    assert payload["finetuning_type"] == "lora"
    assert payload["quantization_bit"] == 4
    assert payload["double_quantization"] is True
    assert payload["quantization_type"] == "nf4"
    assert payload["lora_rank"] == 8
    assert payload["max_steps"] == 3


def test_build_yaml_roundtrip():
    config = QLoRAConfig(model_name_or_path="Qwen/Qwen3-0.6B-Base")
    text = build_yaml(config)
    payload = yaml.safe_load(text)
    assert payload["model_name_or_path"] == "Qwen/Qwen3-0.6B-Base"
    assert payload["lora_target"]


def test_target_modules_detected_from_architecture():
    modules = detect_target_modules(architectures=["Qwen3ForCausalLM"])
    assert "q_proj" in modules and "gate_proj" in modules and "down_proj" in modules
    modules2 = detect_target_modules(_FakeConfig())
    assert "k_proj" in modules2 and "v_proj" in modules2


def test_unknown_architecture_refuses_to_guess():
    with pytest.raises(ValueError, match="no target-module mapping"):
        detect_target_modules(architectures=["UnknownForCausalLM"])


def test_validate_target_modules():
    ok, unknown = validate_target_modules(["q_proj", "k_proj"], ["Qwen3ForCausalLM"])
    assert ok and unknown == []
    ok, unknown = validate_target_modules(["made_up_proj"], ["Qwen3ForCausalLM"])
    assert not ok and unknown == ["made_up_proj"]


def test_prepare_dataset_dir_writes_expected_files(tmp_path):
    rows = [{"instruction": "Translate", "input": "schema", "output": "<sql>SELECT 1</sql>"}]
    target = prepare_dataset_dir(rows, tmp_path / "sft", alias="querydistill")
    train = json.loads((target / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert train["instruction"] == "Translate"
    info = json.loads((target / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["querydistill"]["file_name"] == "train.jsonl"
    assert info["querydistill"]["columns"]["prompt"] == "instruction"


def test_build_command_uses_cli():
    command = build_command(QLoRAConfig(), "config.yaml")
    assert command[-2:] == ["train", "config.yaml"]


def test_run_dry_run_writes_yaml_without_executing(tmp_path):
    config = QLoRAConfig(output_dir=str(tmp_path / "out"), logging_dir=str(tmp_path / "log"))
    report = run_llamafactory(
        config, tmp_path / "train.yaml", tmp_path / "trainer.log", dry_run=True
    )
    assert report["dry_run"] is True
    assert (tmp_path / "train.yaml").exists()
    payload = yaml.safe_load((tmp_path / "train.yaml").read_text(encoding="utf-8"))
    assert payload["stage"] == "sft"


def test_collect_outputs_missing_dir():
    from querydistill.training.llamafactory_backend import collect_outputs

    report = collect_outputs("/nonexistent/checkpoint-dir")
    assert report["exists"] is False
