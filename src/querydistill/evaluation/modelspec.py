"""ModelSpec and model identity helpers for formal evaluation.

Formal evaluation compares Base / Gold-SFT / Distilled-SFT / GRPO / Merged /
GPTQ artifacts. A ModelSpec makes the stage and the exact source artifact
explicit instead of assuming ``AutoModelForCausalLM.from_pretrained(path)``
works for every directory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelSpec:
    stage: str
    base_model_path: str | None = None
    adapter_path: str | None = None
    merged_model_path: str | None = None
    quantized_model_path: str | None = None
    quantization: dict | None = None
    artifact_manifest: dict = field(default_factory=dict)
    hash: str | None = None

    def primary_path(self) -> str | None:
        if self.stage == "adapter" and self.adapter_path:
            return self.adapter_path
        if self.stage in {"merged", "base"} and self.merged_model_path:
            return self.merged_model_path
        if self.stage == "gptq" and self.quantized_model_path:
            return self.quantized_model_path
        return self.base_model_path

    def as_dict(self) -> dict:
        return asdict(self)

    def identity(self) -> dict:
        return {
            "stage": self.stage,
            "base_model": self.base_model_path,
            "adapter": self.adapter_path,
            "merged_model": self.merged_model_path,
            "quantized_model": self.quantized_model_path,
            "quantization": self.quantization,
            "artifact_manifest": self.artifact_manifest,
            "hash": self.hash,
        }


def infer_model_spec(model_path: str | Path, stage: str = "evaluation") -> ModelSpec:
    """Best-effort ModelSpec from a single path.

    This is metadata-only; it does not load weights. Use :func:`load_model` to
    actually load a model according to the stage.
    """
    model_path = str(Path(model_path).resolve())
    return ModelSpec(
        stage=stage,
        base_model_path=model_path if stage == "base" else None,
        adapter_path=model_path if stage == "adapter" else None,
        merged_model_path=model_path if stage == "merged" else None,
        quantized_model_path=model_path if stage == "gptq" else None,
    )


def load_model(spec: ModelSpec, **kwargs):
    """Load a model and tokenizer according to the explicit ModelSpec stage.

    Supported stages:
      - ``base``: AutoModelForCausalLM from base_model_path
      - ``adapter``: base model + PeftModel.from_pretrained(adapter_path)
      - ``merged``: AutoModelForCausalLM from merged_model_path
      - ``gptq``: AutoModelForCausalLM from quantized_model_path (GPTQ checkpoint)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if spec.stage == "base":
        if not spec.base_model_path:
            raise ValueError("base ModelSpec requires base_model_path")
        model_path = spec.base_model_path
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    elif spec.stage == "adapter":
        if not spec.base_model_path or not spec.adapter_path:
            raise ValueError("adapter ModelSpec requires base_model_path and adapter_path")
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(spec.base_model_path, **kwargs)
        model = PeftModel.from_pretrained(base, spec.adapter_path)
        model_path = spec.adapter_path
    elif spec.stage == "merged":
        if not spec.merged_model_path:
            raise ValueError("merged ModelSpec requires merged_model_path")
        model_path = spec.merged_model_path
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    elif spec.stage == "gptq":
        if not spec.quantized_model_path:
            raise ValueError("gptq ModelSpec requires quantized_model_path")
        model_path = spec.quantized_model_path
        load_kwargs = dict(kwargs)
        if spec.quantization:
            load_kwargs.setdefault("quantization_config", spec.quantization)
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    else:
        raise ValueError(f"unsupported ModelSpec stage: {spec.stage!r}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    return model, tokenizer
