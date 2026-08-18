"""vLLM serving config and benchmark helper tests."""

from __future__ import annotations

import json

import pytest

from querydistill.serving.vllm import (
    VLLMServeConfig,
    _percentile,
    benchmark,
    build_server_command,
    check_compatibility,
)


def _model_dir(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
    return model_dir


def test_serve_config_validation(tmp_path):
    missing = VLLMServeConfig(model_path=str(tmp_path / "nope"))
    assert missing.validate()
    good = VLLMServeConfig(model_path=str(_model_dir(tmp_path)))
    assert good.validate() == []


def test_build_server_command_flags(tmp_path):
    config = VLLMServeConfig(
        model_path=str(_model_dir(tmp_path)),
        port=8001,
        gpu_memory_utilization=0.7,
        max_model_len=1024,
        enforce_eager=True,
    )
    command = build_server_command(config)
    joined = " ".join(command)
    assert "--port 8001" in joined
    assert "--gpu-memory-utilization 0.7" in joined
    assert "--max-model-len 1024" in joined
    assert "--enforce-eager" in joined
    assert config.model_path in command


def test_compatibility_report_does_not_fake_pass(tmp_path):
    report = check_compatibility(_model_dir(tmp_path))
    assert report["vllm_version"] in {"not installed"} or isinstance(report["vllm_version"], str)
    assert report["load_success"] is None
    assert report["kernel_selected"] is None


def test_benchmark_rejects_bad_input():
    with pytest.raises(ValueError):
        benchmark([], concurrency=1)
    with pytest.raises(ValueError):
        benchmark(["p"], concurrency=0)


def test_percentile_helper():
    assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == 3.0
    assert _percentile([], 50) == 0.0
    assert _percentile([5.0], 95) == 5.0
