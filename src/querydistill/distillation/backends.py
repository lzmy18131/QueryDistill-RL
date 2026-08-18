"""Teacher backend protocol and implementations.

Fail-closed design: real teacher backends receive a ``RealModelContext`` that
contains only example_id / db_id / split (never gold SQL or gold results). The
gold-oracle mock is a test-only double and reads gold from its own map, not
from the model context.
"""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from typing import Any, Protocol, runtime_checkable

from ..outputs.context import RealModelContext  # noqa: F401 - re-exported for callers


@dataclass
class TeacherConfig:
    """Configuration/provenance for a real Transformers teacher.

    This is intentionally separate from the mock teacher. A real teacher must
    never be recorded as ``mock-teacher-*``.
    """

    model_id: str = "Qwen/Qwen3-4B"
    revision: str = "main"
    device_map: str = "auto"
    load_in_4bit: bool = True
    quant_type: str = "nf4"
    compute_dtype: str = "bfloat16"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    prompt_version: str = "v1"

    def backend_kwargs(self) -> dict:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "device_map": self.device_map,
            "load_in_4bit": self.load_in_4bit,
            "quant_type": self.quant_type,
            "compute_dtype": self.compute_dtype,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
        }

    def generation_config(self) -> dict:
        return {
            "load_in_4bit": self.load_in_4bit,
            "quant_type": self.quant_type,
            "compute_dtype": self.compute_dtype,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
        }

    def provenance(self) -> dict:
        return asdict(self)


@runtime_checkable
class TeacherBackend(Protocol):
    name: str

    def generate(
        self, prompt: str, context: dict[str, Any] | None = None, num_candidates: int = 1
    ) -> list[str]:
        """Return ``num_candidates`` candidate SQL completions for one prompt."""
        ...

    def unload(self) -> None:
        """Release GPU memory. Must be called before Student training."""


class MockTeacherBackend:
    """Deterministic test-only mock (never reported as a real teacher).

    ``strategy="gold"`` needs ``gold_oracle``, a standalone example_id -> gold
    SQL map. Gold never enters the ``context`` argument.
    """

    name = "mock-teacher-1.0"

    def __init__(
        self,
        strategy: str = "gold",
        constant_sql: str = "SELECT 1 AS x",
        gold_oracle: dict[str, str] | None = None,
    ):
        if strategy not in {"gold", "constant"}:
            raise ValueError("strategy must be 'gold' or 'constant'")
        self.strategy = strategy
        self.constant_sql = constant_sql
        self.gold_oracle = gold_oracle or {}
        self.loaded = False

    def generate(
        self, prompt: str, context: dict[str, Any] | None = None, num_candidates: int = 1
    ) -> list[str]:
        self.loaded = True
        if num_candidates < 1:
            raise ValueError("num_candidates must be >= 1")
        context = context or {}
        if self.strategy == "constant":
            sql = self.constant_sql
        else:
            example_id = str(context.get("example_id", ""))
            sql = self.gold_oracle.get(example_id, "")
            if not sql:
                raise ValueError(
                    "mock strategy 'gold' requires gold_oracle[example_id]; "
                    "gold must never come from the model context"
                )
        completion = f"<plan>\ntables: from schema\n</plan>\n<sql>\n{sql}\n</sql>"
        return [completion] * num_candidates

    def unload(self) -> None:
        self.loaded = False


class TransformersTeacherBackend:
    """Real teacher via HuggingFace Transformers (offline, then unload).

    Default for the 8GB machine is 4-bit inference. Not downloaded in this
    round; config + dry-run + unit tests only.
    """

    name = "transformers-teacher"

    def __init__(
        self,
        model_id: str,
        revision: str = "main",
        device_map: str = "auto",
        load_in_4bit: bool = True,
        quant_type: str = "nf4",
        compute_dtype: str = "bfloat16",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
    ):
        self.model_id = model_id
        self.revision = revision
        self.device_map = device_map
        self.load_in_4bit = load_in_4bit
        self.quant_type = quant_type
        self.compute_dtype = compute_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self._model: Any = None
        self._tokenizer: Any = None

    def provenance(self) -> dict:
        return {
            "teacher_model": self.model_id,
            "teacher_revision": self.revision,
            "load_in_4bit": self.load_in_4bit,
            "quant_type": self.quant_type,
            "compute_dtype": self.compute_dtype,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
        }

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self.compute_dtype, torch.float16)
        quantization_config = None
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.quant_type,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            torch_dtype=dtype,
            device_map=self.device_map,
            quantization_config=quantization_config,
        )
        self._model.eval()

    def generate(
        self, prompt: str, context: dict[str, Any] | None = None, num_candidates: int = 1
    ) -> list[str]:
        # Real teacher context is safe-only; assert so it can never regress.
        if context is not None:
            unsafe = set(context) - RealModelContext.safe_keys()
            if unsafe:
                raise ValueError(f"teacher backend received unsafe context keys: {sorted(unsafe)}")
        self.load()
        import torch

        inputs = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to(self._model.device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            num_return_sequences=num_candidates,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        texts = [
            self._tokenizer.decode(
                sequence[inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )
            for sequence in outputs
        ]
        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return texts

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - torch may be absent
            pass
