"""TRL GRPOTrainer integration (round-2 hardened).

Hardening:
* GRPO real runs must initialize from an SFT artifact (adapter or merged
  model); base-only initialization is refused for the experiment chain.
* The initialization source is recorded in ``initialization_manifest.json``
  with sha256 + a trainable-parameter fingerprint before/after training.
* Dataset rows carry example_id/db_id; reward identity is never raw-prompt-key.
* Every exposed config field is actually wired (gradient checkpointing,
  use_vllm, lora target modules, prompt token budget).
* The reward function uses ``CompositeReward.score_once`` - each candidate SQL
  executes exactly once; gold results use the persistent cache.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from ..artifacts.manifest import ArtifactManifest, config_hash, hash_or_none
from ..data.dataset import build_prompt_rows
from ..data.schema import Example, load_examples
from ..data.split_policy import SplitPolicy, assert_training_splits
from ..rewards.composite import CompositeReward, GoldResultCache, RewardTrace
from ..sql.environment import SQLExecutionEnvironment
from ..utils import (
    atomic_write_json,
    atomic_write_text,
    load_json,
    sha256_file,
    utc_now_iso,
)
from .callbacks import RewardSampleLogger, write_smoke_readme

_STATUS_FILE = "status.json"


class GRPOBlockedError(RuntimeError):
    pass


class GRPOTrainerNotInstalledError(RuntimeError):
    pass


class GRPOInitializationError(RuntimeError):
    pass


@dataclass
class GRPOSmokeConfig:
    base_model_path: str = "models/qwen3-0.6b-base"
    init_adapter_path: str | None = None
    init_merged_model_path: str | None = None
    examples_path: str = "data/tiny_sql/examples.jsonl"
    registry_path: str = "tests/fixtures/tiny_sql/db_registry.json"
    output_dir: str = "artifacts/smoke/grpo"
    max_samples: int | None = 8
    num_generations: int = 2
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: int = 2
    learning_rate: float = 1e-5
    max_prompt_length: int = 256
    max_completion_length: int = 96
    temperature: float = 1.0
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    bf16: bool = True
    quantize_4bit: bool = True
    use_gradient_checkpointing: bool = False
    use_vllm: bool = False
    report_to: str = "none"
    logging_steps: int = 1
    save_steps: int = 1
    seed: int = 3407
    require_plan: bool = False
    steps_per_generation: int | None = None
    dry_run: bool = False
    resume: bool = False
    run_id: str = ""
    debug_full_trace: bool = False
    allowed_splits: frozenset[str] = frozenset({"train"})

    def validate(self, dry_run: bool = False) -> list[str]:
        problems: list[str] = []
        if not self.base_model_path:
            problems.append("base_model_path is required")
        if self.num_generations < 1:
            problems.append("num_generations must be >= 1")
        if self.max_steps < 1:
            problems.append("max_steps must be >= 1")
        if self.per_device_train_batch_size < 1:
            problems.append("per_device_train_batch_size must be >= 1")
        if self.max_completion_length < 8:
            problems.append("max_completion_length must be >= 8")
        if not self.init_adapter_path and not self.init_merged_model_path:
            problems.append(
                "GRPO must initialize from an SFT artifact: set init_adapter_path "
                "or init_merged_model_path (base-only GRPO is refused)"
            )
        if self.init_adapter_path and self.init_merged_model_path:
            problems.append("set exactly one of init_adapter_path / init_merged_model_path")
        if not dry_run:
            init = self.init_adapter_path or self.init_merged_model_path
            if init and not Path(init).exists():
                problems.append(f"initialization artifact does not exist: {init}")
            if not Path(self.base_model_path).exists():
                problems.append(f"base model does not exist: {self.base_model_path}")
        return problems


def _parameter_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if not parameter.requires_grad:
            continue
        try:
            value_sum = float(parameter.detach().float().sum().item())
        except Exception:  # noqa: BLE001 - fingerprint is best effort
            value_sum = 0.0
        digest.update(f"{name}|{parameter.numel()}|{parameter.dtype}|{value_sum:.6f}".encode())
    return digest.hexdigest()


def _trainable_param_snapshot(model) -> dict[str, object]:
    """Return CPU clones of trainable parameters only (no optimizer state)."""
    snapshot: dict[str, object] = {}
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if parameter.requires_grad:
            snapshot[name] = parameter.detach().float().cpu().clone()
    return snapshot


def _trainable_param_sha256(snapshot: dict[str, object]) -> str:

    digest = hashlib.sha256()
    for name in sorted(snapshot):
        tensor = snapshot[name]
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _parameter_delta(snapshot_before: dict[str, object], model) -> dict:

    total_sq = 0.0
    max_abs = 0.0
    changed = 0
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if not parameter.requires_grad or name not in snapshot_before:
            continue
        before = snapshot_before[name]
        after = parameter.detach().float().cpu()
        delta = after - before
        total_sq += float((delta * delta).sum().item())
        abs_delta = float(delta.abs().max().item()) if delta.numel() else 0.0
        max_abs = max(max_abs, abs_delta)
        if abs_delta > 0.0:
            changed += 1
    return {
        "parameter_delta_l2": round(float(total_sq**0.5), 8),
        "max_abs_parameter_delta": round(float(max_abs), 8),
        "changed_parameter_tensor_count": changed,
    }


class SQLRewardFunction:
    """TRL-compatible reward callable with metadata-based identity mapping."""

    def __init__(self, composite: CompositeReward, registry: dict[str, Example]):
        self.composite = composite
        self.registry = registry
        self.traces: list[RewardTrace] = []

    def _example_for(self, index: int, prompts: list[str], kwargs: dict) -> Example:
        example_ids = kwargs.get("example_id")
        if example_ids is not None and index < len(example_ids):
            example_id = str(example_ids[index])
            if example_id in self.registry:
                return self.registry[example_id]
        prompt = prompts[index % len(prompts)] if prompts else ""
        # Fallback for older call sites: locate by prompt text (identity still
        # resolved to example_id by the caller's registry).
        from ..outputs.prompting import build_prompt

        for example in self.registry.values():
            if (
                build_prompt(
                    example.question,
                    example.schema_text,
                    example.db_id,
                    include_plan=False,
                )
                == prompt
            ):
                return example
        raise KeyError(f"no example metadata for rollout {index}")

    def clear_traces(self) -> None:
        self.traces = []

    def __call__(self, completions, prompts=None, **kwargs) -> list[float]:
        prompts = list(prompts or [])
        rewards: list[float] = []
        # Keep only the latest batch so a long GRPO run never accumulates an
        # unbounded in-memory trace list. RewardSampleLogger consumes and then
        # clears this buffer after each call.
        self.traces = []
        for index, completion in enumerate(completions):
            example = self._example_for(index, prompts, kwargs)
            text = completion if isinstance(completion, str) else str(completion)
            trace = self.composite.score_once(example, text)
            self.traces.append(trace)
            rewards.append(trace.breakdown.total)
        return rewards


def _config_payload(config: GRPOSmokeConfig) -> dict:
    payload = asdict(config)
    payload["allowed_splits"] = sorted(config.allowed_splits)
    return payload


_RUNTIME_ONLY_FIELDS = frozenset({"resume", "dry_run", "run_id"})


def _training_config_payload(config: GRPOSmokeConfig) -> dict:
    """Config payload that defines training semantics (runtime fields removed)."""
    payload = _config_payload(config)
    for key in _RUNTIME_ONLY_FIELDS:
        payload.pop(key, None)
    return payload


class GRPOSmokeRunner:
    def __init__(self, config: GRPOSmokeConfig, environment: SQLExecutionEnvironment | None = None):
        self.config = config
        self.environment = environment or SQLExecutionEnvironment.from_registry(
            config.registry_path
        )
        self.output_dir = Path(config.output_dir)
        self.run_id = config.run_id or uuid.uuid4().hex
        self.examples = self._load_examples()
        self.split_report = self._split_report

    def _load_examples(self) -> list[Example]:
        assert_training_splits(self.config.allowed_splits, policy_name="grpo")
        policy = SplitPolicy(
            allowed_splits=set(self.config.allowed_splits), policy_name="train_only"
        )
        examples, report = policy.apply(load_examples(Path(self.config.examples_path)))
        self._split_report = report.as_dict()
        if self.config.max_samples is not None:
            examples = examples[: max(0, int(self.config.max_samples))]
        if not examples:
            raise ValueError("GRPO smoke requires at least one train example")
        return examples

    def _run_identity(self) -> dict:
        init = self.config.init_adapter_path or self.config.init_merged_model_path
        return {
            "run_id": self.run_id,
            "config_hash": config_hash(_training_config_payload(self.config)),
            "dataset_sha256": sha256_file(Path(self.config.examples_path)),
            "base_model": str(Path(self.config.base_model_path).resolve()),
            "init_artifact": str(Path(init).resolve()) if init else None,
        }

    def _ensure_run_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        identity_path = self.output_dir / "run_identity.json"
        if any(self.output_dir.iterdir()):
            if not self.config.resume:
                raise FileExistsError(
                    f"output dir is not empty: {self.output_dir}; pass --resume "
                    "only to continue an identical run"
                )
            if not identity_path.exists():
                raise RuntimeError(
                    f"resume requested but {identity_path} is missing; cannot verify run identity"
                )
            stored = load_json(identity_path)
            # Resume must inherit the original run_id instead of minting a new one.
            self.run_id = stored.get("run_id", self.run_id)
            expected = self._run_identity()
            mismatches = []
            for key in ("config_hash", "dataset_sha256", "base_model", "init_artifact"):
                if stored.get(key) != expected[key]:
                    mismatches.append(
                        f"{key}: stored={stored.get(key)!r} current={expected[key]!r}"
                    )
            if mismatches:
                raise RuntimeError(
                    "run identity mismatch; refusing resume: " + "; ".join(mismatches)
                )
        else:
            atomic_write_json(identity_path, self._run_identity())

    def _resume_checkpoint(self) -> str | None:
        if not self.config.resume:
            return None
        trainer_dir = self.output_dir / "trainer"
        checkpoints = sorted(
            trainer_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if not checkpoints:
            raise GRPOInitializationError(
                "--resume requested but no trainer checkpoint exists; "
                "refusing to silently restart from the SFT adapter"
            )
        return str(checkpoints[-1])

    def reward_environment(self) -> CompositeReward:
        dataset_hash = sha256_file(Path(self.config.examples_path))
        database_fingerprints = {
            db_id: sha256_file(path) for db_id, path in self.environment.db_paths.items()
        }
        cache = GoldResultCache(
            cache_dir=self.output_dir / "gold_cache",
            dataset_hash=dataset_hash,
            database_fingerprints=database_fingerprints,
        )
        return CompositeReward(
            environment=self.environment,
            require_plan=self.config.require_plan,
            gold_cache=cache,
        )

    def run(self, dry_run: bool | None = None) -> dict:
        dry_run = self.config.dry_run if dry_run is None else dry_run
        problems = self.config.validate(dry_run=dry_run)
        if problems:
            raise ValueError("invalid GRPO config: " + "; ".join(problems))
        self._ensure_run_dir()

        environment_snapshot = self.environment.registry_snapshot()
        resolved = {
            "run_id": self.run_id,
            "config": _config_payload(self.config),
            "environment": environment_snapshot,
            "split_report": self.split_report,
        }
        atomic_write_json(self.output_dir / "environment.json", resolved)
        atomic_write_text(
            self.output_dir / "resolved_config.yaml",
            yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
        )

        if dry_run:
            write_smoke_readme(
                self.output_dir / "README.md",
                "GRPO real-reward smoke",
                "DRY_RUN",
                {
                    "examples": len(self.examples),
                    "split_report": self.split_report,
                    "init_adapter_path": self.config.init_adapter_path,
                    "init_merged_model_path": self.config.init_merged_model_path,
                    "real_sqlite_reward": True,
                },
            )
            atomic_write_json(
                self.output_dir / _STATUS_FILE,
                {
                    "run_id": self.run_id,
                    "status": "DRY_RUN",
                    "real_sqlite_reward": True,
                    "trained": False,
                    "split_report": self.split_report,
                },
            )
            return {"status": "DRY_RUN", "examples": len(self.examples)}

        return self._train()

    def _train(self) -> dict:
        try:
            import torch
        except Exception as exc:  # noqa: BLE001
            raise GRPOTrainerNotInstalledError(f"torch unavailable: {exc}") from exc
        if not torch.cuda.is_available():
            self._record_blocked("GRPO_SMOKE_BLOCKED_NO_CUDA", "torch.cuda.is_available() is False")
            raise GRPOBlockedError("GRPO smoke requires CUDA")

        try:
            from datasets import Dataset
            from peft import LoraConfig, PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from trl import GRPOConfig, GRPOTrainer
        except Exception as exc:  # noqa: BLE001
            self._record_blocked(
                "GRPO_SMOKE_BLOCKED_MISSING_DEPENDENCY", f"training dependencies missing: {exc}"
            )
            raise GRPOTrainerNotInstalledError(
                "GRPO smoke requires transformers/trl/peft/datasets installed"
            ) from exc

        dtype = (
            torch.bfloat16
            if (self.config.bf16 and torch.cuda.is_bf16_supported())
            else torch.float16
        )
        quantization_config = None
        if self.config.quantize_4bit:
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                )
            except Exception as exc:  # noqa: BLE001
                self._record_blocked("GRPO_SMOKE_BLOCKED_BNB", f"4bit unavailable: {exc}")
                raise GRPOBlockedError(f"4-bit quantization unavailable: {exc}") from exc

        try:
            if self.config.init_adapter_path and Path(self.config.init_adapter_path).exists():
                tokenizer = AutoTokenizer.from_pretrained(self.config.init_adapter_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_path)
        except Exception as exc:  # noqa: BLE001
            self._record_blocked("GRPO_SMOKE_BLOCKED_TOKENIZER", f"tokenizer load failed: {exc}")
            raise GRPOBlockedError(f"tokenizer load failed: {exc}") from exc

        rows, registry = build_prompt_rows(
            self.examples,
            include_plan=self.config.require_plan,
            tokenizer=tokenizer,
            max_prompt_tokens=self.config.max_prompt_length,
        )
        dataset = Dataset.from_list(rows)

        init_manifest: dict = {}
        peft_config = None
        try:
            if self.config.init_adapter_path:
                adapter_path = Path(self.config.init_adapter_path)
                base = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model_path,
                    torch_dtype=dtype,
                    quantization_config=quantization_config,
                    device_map={"": 0} if quantization_config is None else "auto",
                )
                model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=True)
                init_manifest = {
                    "initialization_source": "sft_adapter",
                    "adapter_path": str(adapter_path.resolve()),
                    "adapter_sha256": sha256_file(adapter_path / "adapter_model.safetensors"),
                    "base_model": str(Path(self.config.base_model_path).resolve()),
                    "base_config_sha256": sha256_file(
                        Path(self.config.base_model_path) / "config.json"
                    ),
                }
            else:
                merged_path = Path(self.config.init_merged_model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    str(merged_path),
                    torch_dtype=dtype,
                    quantization_config=quantization_config,
                    device_map={"": 0} if quantization_config is None else "auto",
                )
                target_modules = self._resolve_target_modules(model)
                peft_config = LoraConfig(
                    task_type="CAUSAL_LM",
                    r=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=target_modules,
                )
                init_manifest = {
                    "initialization_source": "merged_sft_model",
                    "merged_model_path": str(merged_path.resolve()),
                    "base_model": str(Path(self.config.base_model_path).resolve()),
                    "base_config_sha256": sha256_file(
                        Path(self.config.base_model_path) / "config.json"
                    ),
                }
        except Exception as exc:  # noqa: BLE001
            self._record_blocked(
                "GRPO_SMOKE_BLOCKED_INITIALIZATION", f"init from SFT failed: {exc}"
            )
            raise GRPOInitializationError(
                f"failed to initialize GRPO from SFT state: {exc}"
            ) from exc

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Qwen chat checkpoints use <|im_end|> as the real end-of-turn token.
        # Base tokenizers often default to <|endoftext|>, which would make the
        # model keep generating after finishing one <sql> block.
        if getattr(tokenizer, "eos_token_id", None) is not None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id

        init_manifest["trainable_parameter_fingerprint_before"] = _parameter_fingerprint(model)
        param_snapshot_before = _trainable_param_snapshot(model)
        init_manifest["trainable_param_sha256_before"] = _trainable_param_sha256(
            param_snapshot_before
        )
        init_manifest["created_at"] = utc_now_iso()
        atomic_write_json(self.output_dir / "initialization_manifest.json", init_manifest)

        reward_fn = SQLRewardFunction(self.reward_environment(), registry)
        logger = RewardSampleLogger(
            reward_fn,
            self.output_dir / "reward_samples.jsonl",
            registry,
            debug_full_trace=self.config.debug_full_trace,
            run_id=self.run_id,
        )

        # TRL >= 1.10 forbids setting both generation_batch_size and
        # steps_per_generation.  When steps_per_generation is configured, let
        # TRL derive generation_batch_size from it.
        generation_batch_size = (
            None
            if self.config.steps_per_generation is not None
            else (
                self.config.per_device_train_batch_size
                * self.config.gradient_accumulation_steps
                * self.config.num_generations
            )
        )
        training_args = GRPOConfig(
            output_dir=str(self.output_dir / "trainer"),
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_generations=self.config.num_generations,
            generation_batch_size=generation_batch_size,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            report_to=self.config.report_to,
            bf16=self.config.bf16 and torch.cuda.is_bf16_supported(),
            fp16=not (self.config.bf16 and torch.cuda.is_bf16_supported()),
            gradient_checkpointing=self.config.use_gradient_checkpointing,
            use_vllm=self.config.use_vllm,
            max_completion_length=self.config.max_completion_length,
            temperature=self.config.temperature,
            seed=self.config.seed,
            steps_per_generation=self.config.steps_per_generation,
            disable_tqdm=False,
        )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[logger],
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

        resume_from_checkpoint = self._resume_checkpoint()

        try:
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        except torch.cuda.OutOfMemoryError as exc:
            self._record_blocked(
                "GRPO_SMOKE_OOM",
                "CUDA OOM during GRPO smoke; reducing size requires a later round",
            )
            raise GRPOBlockedError(
                "GRPO smoke hit CUDA OOM; recorded as GRPO_SMOKE_OOM (not a pass)"
            ) from exc

        adapter_dir = self.output_dir / "adapter"
        trainer.save_model(str(adapter_dir))
        metrics = list(trainer.state.log_history)
        global_step = int(getattr(trainer.state, "global_step", len(metrics)))
        fingerprint_after = _parameter_fingerprint(model)
        param_snapshot_after = _trainable_param_snapshot(model)
        param_sha_after = _trainable_param_sha256(param_snapshot_after)
        delta_evidence = _parameter_delta(param_snapshot_before, model)
        evidence = self._learning_signal(
            metrics, init_manifest, fingerprint_after, logger.count, global_step=global_step
        )
        evidence["split_report"] = self.split_report
        evidence["trainable_parameter_fingerprint_after"] = fingerprint_after
        evidence["trainable_parameter_fingerprint_changed"] = (
            init_manifest["trainable_parameter_fingerprint_before"] != fingerprint_after
        )
        evidence["trainable_param_sha256_before"] = init_manifest["trainable_param_sha256_before"]
        evidence["trainable_param_sha256_after"] = param_sha_after
        evidence["trainable_param_sha256_changed"] = (
            init_manifest["trainable_param_sha256_before"] != param_sha_after
        )
        evidence.update(delta_evidence)
        parameter_delta_path = self.output_dir / "parameter_delta.json"
        atomic_write_json(
            parameter_delta_path,
            {
                "trainable_param_sha256_before": init_manifest["trainable_param_sha256_before"],
                "trainable_param_sha256_after": param_sha_after,
                "changed": init_manifest["trainable_param_sha256_before"] != param_sha_after,
                **delta_evidence,
            },
        )

        atomic_write_json(self.output_dir / "metrics.json", {"log_history": metrics})
        atomic_write_text(
            self.output_dir / "trainer_log",
            "\n".join(str(entry) for entry in metrics) + "\n",
        )
        atomic_write_json(self.output_dir / _STATUS_FILE, evidence)
        write_smoke_readme(
            self.output_dir / "README.md",
            "GRPO real-reward smoke",
            evidence["status"],
            evidence,
        )
        ArtifactManifest(
            stage="grpo",
            input_artifact=init_manifest.get("adapter_path")
            or init_manifest.get("merged_model_path"),
            input_sha256=init_manifest.get("adapter_sha256")
            or hash_or_none(init_manifest.get("merged_model_path")),
            output_artifact=str(adapter_dir.resolve()),
            base_model=str(Path(self.config.base_model_path).resolve()),
            adapter=str(adapter_dir.resolve()),
            config_hash=config_hash(_config_payload(self.config)),
            extra={"learning_signal": evidence["learning_signal"]},
        ).write(self.output_dir)
        return evidence

    def _resolve_target_modules(self, model) -> list[str]:
        from ..training.llamafactory_backend import detect_target_modules

        if self.config.lora_target_modules:
            return list(self.config.lora_target_modules)
        try:
            return detect_target_modules(model.config)
        except Exception as exc:  # noqa: BLE001
            raise GRPOInitializationError(
                f"could not detect LoRA target modules from the loaded model: {exc}"
            ) from exc

    def _learning_signal(
        self,
        metrics: list[dict],
        init_manifest: dict,
        fingerprint_after: str,
        reward_samples_logged: int,
        global_step: int | None = None,
    ) -> dict:
        reward_stds = [
            float(entry.get("rewards/RewardSampleLogger/std", 0.0))
            for entry in metrics
            if "rewards/RewardSampleLogger/std" in entry
        ]
        grad_norms = [
            float(entry.get("grad_norm", 0.0)) for entry in metrics if "grad_norm" in entry
        ]
        has_variance = any(std > 0 for std in reward_stds)
        has_update = any(abs(grad) > 0 for grad in grad_norms) or (
            init_manifest.get("trainable_parameter_fingerprint_before") != fingerprint_after
        )
        learning_signal = (
            "GRPO_LEARNING_SIGNAL_PASS"
            if (has_variance and has_update)
            else "GRPO_LEARNING_SIGNAL_INSUFFICIENT"
        )
        return {
            "run_id": self.run_id,
            "status": "GRPO_INTEGRATION_PASS",
            "learning_signal": learning_signal,
            "trained": True,
            "steps": global_step if global_step is not None else len(metrics),
            "real_sqlite_reward": True,
            "reward_samples_logged": reward_samples_logged,
            "reward_std_values": reward_stds,
            "grad_norm_values": grad_norms,
            "initialization_source": init_manifest.get("initialization_source"),
            "adapter_path": init_manifest.get("adapter_path"),
            "adapter_sha256": init_manifest.get("adapter_sha256"),
            "adapter_dir": str(self.output_dir / "adapter"),
        }

    def _record_blocked(self, status: str, reason: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self.output_dir / _STATUS_FILE,
            {
                "run_id": self.run_id,
                "status": status,
                "reason": reason,
                "trained": False,
                "real_sqlite_reward": True,
            },
        )
        write_smoke_readme(
            self.output_dir / "README.md", "GRPO real-reward smoke", status, {"reason": reason}
        )
