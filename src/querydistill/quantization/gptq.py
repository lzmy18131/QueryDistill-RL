"""GPTQModel INT4 quantization adapter.

Formal flow (this round only smokes the API path):

trained LoRA adapter -> merge into non-quantized base -> save merged checkpoint
-> calibration dataset (train/calibration split ONLY) -> GPTQ INT4 -> deployable
checkpoint.

Test contamination guard: ``build_calibration_dataset`` refuses any example
whose split is not train/calibration, and the runner records the exact
calibration example ids so an auditor can verify no dev/test example was used.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from ..artifacts.manifest import ArtifactManifest, config_hash, hash_or_none
from ..data.dataset import assert_calibration_split, tokenize_calibration
from ..data.schema import load_examples
from ..data.split_policy import SplitPolicy, assert_calibration_splits
from ..utils import atomic_write_json, atomic_write_text

_STATUS_FILE = "status.json"


class GPTQBlockedError(RuntimeError):
    pass


@dataclass
class GPTQConfig:
    base_model_path: str
    adapter_path: str | None = None
    merged_output_dir: str | None = None
    output_dir: str = "artifacts/smoke/gptq"
    calibration_examples_path: str = "data/tiny_sql/examples.jsonl"
    max_calibration_samples: int = 16
    calibration_max_length: int = 512
    bits: int = 4
    group_size: int = 128
    desc_act: bool = False
    sym: bool = True
    batch_size: int = 1
    dry_run: bool = False

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.bits not in {2, 3, 4, 8}:
            problems.append("bits must be one of 2/3/4/8")
        if self.group_size <= 0:
            problems.append("group_size must be positive")
        if self.batch_size < 1:
            problems.append("batch_size must be >= 1")
        if self.max_calibration_samples < 1:
            problems.append("max_calibration_samples must be >= 1")
        return problems


def merge_lora_adapter(
    base_model_path: str | Path, adapter_path: str | Path, output_dir: str | Path
) -> Path:
    """Merge a trained LoRA adapter into the base and save a standalone checkpoint."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model_path = Path(base_model_path)
    adapter_path = Path(adapter_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(str(base_model_path))
    model = PeftModel.from_pretrained(base, str(adapter_path))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(output_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def build_calibration_dataset(
    examples_path: str | Path,
    tokenizer,
    max_samples: int = 16,
    max_length: int = 512,
    allowed_splits: frozenset[str] = frozenset({"train", "calibration"}),
) -> tuple[list[dict], dict]:
    """Tokenized calibration samples from train/calibration splits only.

    A dataset file may contain dev/test rows; they are excluded by SplitPolicy
    and recorded. The contamination guarantee is that every **selected** sample
    belongs to an allowed split (asserted on the selected subset).
    """
    assert_calibration_splits(allowed_splits, policy_name="gptq_calibration")
    policy = SplitPolicy(allowed_splits=set(allowed_splits), policy_name="calibration")
    examples, report = policy.apply(load_examples(Path(examples_path)), source_path=examples_path)
    selected = examples[:max_samples]
    if not selected:
        raise ValueError(f"calibration dataset has no {sorted(allowed_splits)} examples to select")
    assert_calibration_split(selected, allowed=allowed_splits)
    texts = [
        f"Question: {example.question}\nSchema:\n{example.schema_text}" for example in selected
    ]
    return tokenize_calibration(tokenizer, texts, max_length=max_length), report.as_dict()


class GPTQRunner:
    def __init__(self, config: GPTQConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)

    def _record(self, status: str, payload: dict) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {"status": status, **payload}
        payload.setdefault("quantized", False)
        atomic_write_json(self.output_dir / _STATUS_FILE, payload)
        lines = ["# GPTQ INT4 smoke", "", f"STATUS: {status}", ""]
        for key, value in payload.items():
            lines.append(f"- {key}: {value}")
        atomic_write_text(self.output_dir / "README.md", "\n".join(lines) + "\n")
        return payload

    def run(self, dry_run: bool | None = None) -> dict:
        dry_run = self.config.dry_run if dry_run is None else dry_run
        problems = self.config.validate()
        if problems:
            raise ValueError("invalid GPTQ config: " + "; ".join(problems))

        resolved = self.config.__dict__
        atomic_write_text(
            self.output_dir / "resolved_config.yaml",
            yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if dry_run:
            return self._record(
                "DRY_RUN",
                {
                    "calibration_splits_allowed": ["train", "calibration"],
                    "quantized": False,
                    "api_path_verified": False,
                },
            )

        base_path = Path(self.config.base_model_path)
        if not base_path.exists():
            return self._record("GPTQ_SMOKE_BLOCKED_NO_BASE_MODEL", {"reason": str(base_path)})

        merged_dir = base_path
        if self.config.adapter_path:
            adapter_path = Path(self.config.adapter_path)
            if not adapter_path.exists():
                return self._record("GPTQ_SMOKE_BLOCKED_NO_ADAPTER", {"reason": str(adapter_path)})
            merged_dir = Path(self.config.merged_output_dir or (self.output_dir / "merged"))
            try:
                merged_dir = merge_lora_adapter(base_path, adapter_path, merged_dir)
            except Exception as exc:  # noqa: BLE001 - real failure recorded, not faked
                return self._record("GPTQ_SMOKE_BLOCKED_MERGE_FAILED", {"reason": str(exc)})

        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            return self._record("GPTQ_SMOKE_BLOCKED_MISSING_DEPENDENCY", {"reason": str(exc)})

        try:
            tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))
        except Exception as exc:  # noqa: BLE001
            return self._record("GPTQ_SMOKE_BLOCKED_TOKENIZER_LOAD", {"reason": str(exc)})

        calibration_policy = SplitPolicy(
            allowed_splits={"train", "calibration"}, policy_name="calibration"
        )
        calibration, calibration_report = calibration_policy.apply(
            load_examples(Path(self.config.calibration_examples_path)),
            source_path=self.config.calibration_examples_path,
        )
        calibration = calibration[: self.config.max_calibration_samples]
        try:
            assert_calibration_split(calibration)
        except ValueError as exc:
            return self._record("GPTQ_SMOKE_BLOCKED_CONTAMINATION", {"reason": str(exc)})

        try:
            samples, build_report = build_calibration_dataset(
                self.config.calibration_examples_path,
                tokenizer,
                max_samples=self.config.max_calibration_samples,
                max_length=self.config.calibration_max_length,
            )
        except Exception as exc:  # noqa: BLE001
            return self._record("GPTQ_SMOKE_BLOCKED_CALIBRATION", {"reason": str(exc)})

        atomic_write_json(
            self.output_dir / "calibration_manifest.json",
            {
                "example_ids": [example.example_id for example in calibration],
                "splits": [example.split for example in calibration],
                "contamination_check": "all selected splits in {train, calibration}",
                "split_report": calibration_report.as_dict(),
                "tokenized_report": build_report,
            },
        )

        try:
            from gptqmodel import GPTQModel, QuantizeConfig
        except Exception as exc:  # noqa: BLE001
            return self._record("GPTQ_SMOKE_BLOCKED_MISSING_GPTQMODEL", {"reason": str(exc)})

        try:
            quantize_config = QuantizeConfig(
                bits=self.config.bits,
                group_size=self.config.group_size,
                desc_act=self.config.desc_act,
                sym=self.config.sym,
            )
            model = GPTQModel.load(str(merged_dir), quantize_config=quantize_config)
            model.quantize(samples, batch_size=self.config.batch_size)
            model.save(str(self.output_dir / "quantized"))
        except Exception as exc:  # noqa: BLE001 - real failure recorded, not faked
            return self._record("GPTQ_SMOKE_BLOCKED_QUANTIZE_FAILED", {"reason": str(exc)})

        payload = {
            "quantized": True,
            "bits": self.config.bits,
            "group_size": self.config.group_size,
            "desc_act": self.config.desc_act,
            "sym": self.config.sym,
            "source_checkpoint": str(merged_dir),
            "calibration_samples": len(samples),
            "quantized_output": str(self.output_dir / "quantized"),
        }
        ArtifactManifest(
            stage="gptq",
            input_artifact=str(merged_dir.resolve()) if merged_dir.exists() else str(merged_dir),
            input_sha256=hash_or_none(merged_dir / "config.json"),
            output_artifact=str((self.output_dir / "quantized").resolve()),
            base_model=str(Path(self.config.base_model_path).resolve()),
            adapter=(
                str(Path(self.config.adapter_path).resolve()) if self.config.adapter_path else None
            ),
            config_hash=config_hash(asdict(self.config)),
            extra={"calibration_samples": len(samples)},
        ).write(self.output_dir)
        return self._record("PASS", payload)


def check_compatibility(quantized_dir: str | Path) -> dict:
    """Metadata-level GPTQ/vLLM compatibility check (kernel verified only at serve time)."""
    quantized_dir = Path(quantized_dir)
    config_path = quantized_dir / "config.json"
    if not config_path.exists():
        return {"exists": False, "reason": f"{config_path} not found"}

    config = json.loads(config_path.read_text(encoding="utf-8"))
    quant = config.get("quantization_config", {})
    report = {
        "exists": True,
        "model_format": config.get("model_type", "unknown"),
        "quantization_bits": quant.get("bits"),
        "quantization_backend": quant.get("quant_method", quant.get("backend", "unknown")),
        "group_size": quant.get("group_size"),
        "desc_act": quant.get("desc_act", False),
        "sym": quant.get("sym", None),
        "note": (
            "vLLM kernel selection and load success are recorded at serve time "
            "(scripts/serve_vllm.sh). No PASS is claimed here."
        ),
    }
    return report
