"""Hardware doctor tests."""

from __future__ import annotations

import pytest

from querydistill.hardware import detect_wsl, format_report, probe


def test_detect_wsl_returns_pair():
    is_wsl, detail = detect_wsl()
    assert isinstance(is_wsl, bool)
    assert isinstance(detail, str) and detail


def test_probe_has_required_fields(settings):
    report = probe(settings)
    payload = report.as_dict()
    required = [
        "os",
        "wsl_detected",
        "python_version",
        "cpu_model",
        "ram_total_gb",
        "ram_free_gb",
        "gpu_name",
        "gpu_vram_total_mib",
        "nvidia_driver",
        "cuda_available",
        "torch_version",
        "torch_cuda_version",
        "bf16_supported",
        "transformers_version",
        "trl_version",
        "peft_version",
        "bitsandbytes_version",
        "sqlglot_version",
        "gptqmodel_version",
        "vllm_version",
        "project_root",
        "project_free_gb",
        "cache_root",
        "cache_free_gb",
    ]
    missing = [name for name in required if name not in payload]
    assert missing == []
    assert payload["ram_total_gb"] > 0
    assert isinstance(payload["warnings"], list)


def test_format_report_contains_key_rows(settings):
    text = format_report(probe(settings))
    for needle in ("OS", "WSL detection", "Python", "GPU", "CUDA availability", "torch"):
        assert needle in text


@pytest.mark.gpu
def test_gpu_probe_sees_cuda_when_present():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device in this environment")
    assert probe().cuda_available is True
    assert probe().cuda_device_count >= 1
