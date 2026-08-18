"""vLLM serving configuration + OpenAI-compatible benchmark client.

vLLM is the serving/inference backend, never a training framework. Windows
cannot run vLLM natively; the supported path is WSL2/Linux (a separate ``serve``
environment is recommended so its PyTorch/CUDA wheel pins do not damage the
training environment).

First-round rule: if vLLM is installable, run the 0.6B smoke server and exactly
one request. Otherwise the status file records ``VLLM_NOT_SMOKE_VERIFIED`` -
never a fabricated PASS.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from ..quantization.gptq import check_compatibility as _gptq_compatibility
from ..utils import atomic_write_json, package_version


@dataclass
class VLLMServeConfig:
    model_path: str
    host: str = "127.0.0.1"
    port: int = 8000
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int | None = 2048
    dtype: str = "auto"
    max_num_seqs: int = 8
    enforce_eager: bool = False
    trust_remote_code: bool = True
    api_key: str = "querydistill-smoke"

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not Path(self.model_path).exists():
            problems.append(f"model_path does not exist: {self.model_path}")
        if not 0 < self.gpu_memory_utilization <= 1.0:
            problems.append("gpu_memory_utilization must be in (0, 1]")
        if self.tensor_parallel_size < 1:
            problems.append("tensor_parallel_size must be >= 1")
        return problems


def build_server_command(config: VLLMServeConfig) -> list[str]:
    executable = shutil.which("vllm") or shutil.which("vllm.exe")
    base = [executable, "serve"] if executable else ["python", "-m", "vllm", "serve"]
    command = base + [
        config.model_path,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--dtype",
        config.dtype,
        "--max-num-seqs",
        str(config.max_num_seqs),
        "--api-key",
        config.api_key,
    ]
    if config.max_model_len is not None:
        command += ["--max-model-len", str(config.max_model_len)]
    if config.enforce_eager:
        command.append("--enforce-eager")
    if config.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def check_compatibility(model_dir: str | Path) -> dict:
    gptq = _gptq_compatibility(Path(model_dir))
    vllm_version = package_version("vllm")
    return {
        "model_dir": str(model_dir),
        "vllm_version": vllm_version or "not installed",
        "quantization": gptq,
        "kernel_selected": None,
        "load_success": None,
        "fallback_behavior": (
            "not applicable until a vLLM server is actually started; "
            "if the GPTQ kernel is unavailable the smoke is recorded as BLOCKED, not PASS"
        ),
    }


def _sse_chunks(response) -> list[dict]:
    events: list[dict] = []
    for raw in response.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        payload = raw[len("data:") :].strip()
        if payload == "[DONE]":
            break
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return events


def _one_request(
    prompt: str, endpoint: str, model: str, max_tokens: int, api_key: str | None
) -> dict:
    import requests

    url = f"{endpoint}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    started = time.monotonic()
    ttft = None
    output_tokens = 0
    response = requests.post(url, json=payload, headers=headers, stream=True, timeout=300)
    response.raise_for_status()
    for event in _sse_chunks(response):
        if ttft is None:
            ttft = time.monotonic() - started
        choices = event.get("choices") or []
        if choices and choices[0].get("delta", {}).get("content"):
            output_tokens += 1
    total = time.monotonic() - started
    if ttft is None:
        ttft = total
    return {
        "ttft_ms": round(ttft * 1000.0, 2),
        "total_latency_ms": round(total * 1000.0, 2),
        "output_tokens": output_tokens,
    }


def _sample_vram(stop: threading.Event, samples: list[float]) -> None:
    while not stop.is_set():
        try:
            import subprocess

            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                samples.append(float(proc.stdout.strip().splitlines()[0].strip()))
        except Exception:  # noqa: BLE001 - best effort
            pass
        stop.wait(0.2)


def benchmark(
    prompts: list[str],
    endpoint: str = "http://127.0.0.1:8000/v1",
    model: str = "querydistill",
    concurrency: int = 1,
    max_tokens: int = 128,
    api_key: str | None = None,
) -> dict:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if not prompts:
        raise ValueError("at least one prompt is required")

    stop = threading.Event()
    vram_samples: list[float] = []
    sampler = threading.Thread(target=_sample_vram, args=(stop, vram_samples), daemon=True)
    sampler.start()

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_one_request, prompt, endpoint, model, max_tokens, api_key)
            for prompt in prompts
        ]
        results = [future.result() for future in futures]
    wall = time.monotonic() - started
    stop.set()
    sampler.join(timeout=1.0)

    ttfts = [r["ttft_ms"] for r in results]
    totals = [r["total_latency_ms"] for r in results]
    output_tokens = sum(r["output_tokens"] for r in results)

    report = {
        "concurrency": concurrency,
        "requests": len(results),
        "ttft_ms": {"p50": _percentile(ttfts, 50), "p95": _percentile(ttfts, 95)},
        "total_latency_ms": {"p50": _percentile(totals, 50), "p95": _percentile(totals, 95)},
        "tokens_per_sec": round(output_tokens / wall, 2) if wall > 0 else 0.0,
        "requests_per_sec": round(len(results) / wall, 2) if wall > 0 else 0.0,
        "peak_vram_mib": round(max(vram_samples), 1) if vram_samples else None,
        "wall_seconds": round(wall, 2),
        "per_request": results,
    }
    return report


def _percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, int(round((percent / 100.0) * (len(ordered) - 1)))))
    return round(ordered[index], 2)


def save_benchmark(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, report)
    return path
