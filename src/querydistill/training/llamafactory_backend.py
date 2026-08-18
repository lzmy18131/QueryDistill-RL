"""LLaMA-Factory integration layer.

LLaMA-Factory is used as the mature QLoRA SFT pipeline. This module is a thin,
honest wrapper: it generates/validates the project's own YAML, prepares the
project's custom dataset registration, invokes the upstream CLI as a
subprocess, collects output paths / trainer logs, saves the resolved config,
and handles checkpoint resume + error reporting.

It is NOT presented as a contribution of this project; the project's own value
is the data, verification, safety, reward, evaluation and quantization layers.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..utils import atomic_write_json, atomic_write_text, load_jsonl

ARCHITECTURE_TARGET_MODULES: dict[str, list[str]] = {
    "qwen3forcausallm": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "qwen2forcausallm": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "llamaforcausallm": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "mistralforcausallm": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class LLaMAFactoryNotInstalledError(RuntimeError):
    pass


class LLaMAFactoryRunError(RuntimeError):
    def __init__(self, message: str, log_path: Path | None = None):
        self.log_path = log_path
        super().__init__(message)


@dataclass
class QLoRAConfig:
    stage: str = "sft"
    model_name_or_path: str = "Qwen/Qwen3-0.6B-Base"
    template: str = "qwen"
    dataset_alias: str = "querydistill"
    dataset_dir: str = "data/sft"
    # Project-side dataset fields (consumed by the CLI, not by LLaMA-Factory).
    source_examples_path: str = "data/tiny_sql/examples.jsonl"
    distillation_path: str | None = None
    target_mode: str = "gold"  # gold | distilled
    train_splits: list[str] = field(default_factory=lambda: ["train"])
    include_plan: bool = False
    strict_distilled: bool = True
    paired_manifest_path: str | None = None
    output_dir: str = "checkpoints/sft/smoke"
    logging_dir: str = "runs/sft/smoke"
    finetuning_type: str = "lora"
    quantization_bit: int = 4
    double_quantization: bool = True
    quantization_type: str = "nf4"
    compute_dtype: str = "bf16"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    num_train_epochs: float = 1.0
    max_steps: int | None = 2
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.0
    logging_steps: int = 1
    save_steps: int = 10
    save_total_limit: int = 2
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    val_size: float = 0.0
    report_to: str = "none"
    max_seq_length: int = 512
    packing: bool = False
    overwrite_output_dir: bool = True
    resume_from_checkpoint: bool = True

    def resolved_dict(self) -> dict:
        payload = {
            "stage": self.stage,
            "do_train": True,
            "model_name_or_path": self.model_name_or_path,
            "template": self.template,
            "dataset": self.dataset_alias,
            "dataset_dir": self.dataset_dir,
            "finetuning_type": self.finetuning_type,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target": self.lora_target,
            "output_dir": self.output_dir,
            "logging_dir": self.logging_dir,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "gradient_checkpointing": self.gradient_checkpointing,
            "val_size": self.val_size,
            "report_to": self.report_to,
            "cutoff_len": self.max_seq_length,
            "packing": self.packing,
            "overwrite_output_dir": self.overwrite_output_dir,
            # LLaMA-Factory raises on resume=True with an empty output dir.
            "resume_from_checkpoint": (
                self.resume_from_checkpoint and any(Path(self.output_dir).glob("checkpoint-*"))
            ),
        }
        if self.max_steps is not None:
            payload["max_steps"] = self.max_steps
        if self.bf16:
            payload["bf16"] = True
        if self.fp16:
            payload["fp16"] = True
        if self.quantization_bit and self.quantization_bit > 0:
            payload["quantization_bit"] = self.quantization_bit
            payload["double_quantization"] = self.double_quantization
            payload["quantization_type"] = self.quantization_type
        return payload


def detect_target_modules(
    model_id_or_config: str | object | None = None,
    architectures: list[str] | None = None,
    local_files_only: bool = True,
) -> list[str]:
    """Detect LoRA target modules from the real model config (never hard-coded guesses).

    When ``local_files_only`` is true (default) this reads a local config JSON
    without downloading anything; pass an explicit ``architectures`` list for
    offline validation.
    """
    if model_id_or_config is None and not architectures:
        raise ValueError("provide model_id_or_config or architectures")
    arch_names: list[str] = list(architectures or [])
    if hasattr(model_id_or_config, "architectures"):
        arch_names = list(model_id_or_config.architectures)
    elif isinstance(model_id_or_config, str):
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                model_id_or_config, local_files_only=local_files_only
            )
            arch_names = list(getattr(config, "architectures", []))
        except Exception as exc:  # noqa: BLE001 - expected offline in first round
            raise ValueError(
                f"cannot detect model architectures for {model_id_or_config}: {exc}"
            ) from exc

    modules = set()
    for name in arch_names:
        key = name.lower()
        for architecture_key, modules_for_arch in ARCHITECTURE_TARGET_MODULES.items():
            if architecture_key in key:
                modules.update(modules_for_arch)
    if not modules:
        raise ValueError(
            f"no target-module mapping for architectures {arch_names}; "
            "refusing to fall back to hard-coded guesses"
        )
    return sorted(modules)


def validate_target_modules(
    target_modules: list[str], architectures: list[str]
) -> tuple[bool, list[str]]:
    expected = detect_target_modules(architectures=architectures)
    unknown = sorted(set(target_modules) - set(expected))
    return (not unknown), unknown


def prepare_dataset_dir(
    rows: list[dict], target_dir: str | Path, alias: str = "querydistill"
) -> Path:
    """Write Alpaca-style train.json + LLaMA-Factory dataset_info.json."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    train_rows = [
        {
            "instruction": row["instruction"],
            "input": row["input"],
            "output": row["output"],
        }
        for row in rows
    ]
    atomic_write_text(
        target_dir / "train.jsonl",
        "\n".join(json.dumps(row, ensure_ascii=False) for row in train_rows) + "\n",
    )
    dataset_info = {
        alias: {
            "file_name": "train.jsonl",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        }
    }
    atomic_write_json(target_dir / "dataset_info.json", dataset_info)
    return target_dir


def build_yaml(config: QLoRAConfig) -> str:
    payload = config.resolved_dict()
    body = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    return (
        "# Generated by QueryDistill-RL llamafactory_backend.\n"
        "# LLaMA-Factory is used as the upstream QLoRA SFT implementation.\n" + body
    )


def _llamafactory_executable() -> str | None:
    """Prefer the CLI script next to the active interpreter, then PATH."""
    import sys

    sibling = Path(sys.executable).parent / (
        "llamafactory-cli.exe" if os.name == "nt" else "llamafactory-cli"
    )
    if sibling.exists():
        return str(sibling)
    return shutil.which("llamafactory-cli") or shutil.which("llamafactory-cli.exe")


def build_command(config: QLoRAConfig, yaml_path: str | Path) -> list[str]:
    executable = _llamafactory_executable()
    if executable:
        return [executable, "train", str(yaml_path)]
    import sys

    return [sys.executable, "-m", "llamafactory.cli", "train", str(yaml_path)]


def _check_llamafactory_available() -> bool:
    if _llamafactory_executable():
        return True
    try:
        import llamafactory  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def run_llamafactory(
    config: QLoRAConfig,
    yaml_path: str | Path,
    log_path: str | Path,
    dry_run: bool = False,
    env: dict | None = None,
    stdout_path: str | Path | None = None,
    stderr_path: str | Path | None = None,
) -> dict:
    yaml_path = Path(yaml_path)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(yaml_path, build_yaml(config))

    report: dict = {
        "yaml_path": str(yaml_path),
        "log_path": str(log_path),
        "command": build_command(config, yaml_path),
        "dry_run": dry_run,
        "returncode": None,
    }
    if dry_run:
        return report

    if not _check_llamafactory_available():
        raise LLaMAFactoryNotInstalledError(
            "LLaMA-Factory CLI is not installed in this environment. "
            "Install the train extras (or `pip install llamafactory`) and retry."
        )
    if not _llamafactory_executable():
        # The package exists but only exposes `python -m llamafactory.cli`.
        import sys

        try:
            subprocess.run(
                [sys.executable, "-m", "llamafactory.cli", "version"],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
                env=env,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLaMAFactoryNotInstalledError(
                f"`llamafactory-cli` not on PATH and "
                f"`{sys.executable} -m llamafactory.cli` failed: {exc}"
            ) from exc

    if stdout_path is not None and stderr_path is not None:
        stdout_path = Path(stdout_path)
        stderr_path = Path(stderr_path)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        with (
            stdout_path.open("w", encoding="utf-8", newline="\n") as out_handle,
            stderr_path.open("w", encoding="utf-8", newline="\n") as err_handle,
        ):
            process = subprocess.Popen(
                build_command(config, yaml_path),
                stdout=out_handle,
                stderr=err_handle,
                text=True,
                env=env,
            )
            returncode = process.wait()
    else:
        with log_path.open("w", encoding="utf-8", newline="\n") as handle:
            process = subprocess.Popen(
                build_command(config, yaml_path),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            returncode = process.wait()
    report["returncode"] = returncode
    if returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
        raise LLaMAFactoryRunError(
            f"LLaMA-Factory exited with code {returncode}; see {log_path}\n--- log tail ---\n{tail}",
            log_path=log_path,
        )
    return report


def collect_outputs(output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return {
            "output_dir": str(output_dir),
            "exists": False,
            "adapter_config": None,
            "checkpoints": [],
            "trainer_logs": [],
        }
    adapter_config = output_dir / "adapter_config.json"
    checkpoints = sorted(p.as_posix() for p in output_dir.glob("checkpoint-*"))
    trainer_logs = sorted(p.as_posix() for p in output_dir.glob("trainer_log.jsonl"))
    return {
        "output_dir": str(output_dir),
        "exists": True,
        "adapter_config": str(adapter_config) if adapter_config.exists() else None,
        "checkpoints": checkpoints,
        "trainer_logs": trainer_logs,
    }


def collect_trainer_log(output_dir: str | Path) -> list[dict]:
    logs: list[dict] = []
    for path in sorted(Path(output_dir).glob("trainer_log.jsonl")):
        logs.extend(load_jsonl(path))
    return logs
