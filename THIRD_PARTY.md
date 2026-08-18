# Third-party software and attribution

All third-party libraries are used through their official public APIs. No
third-party project source code has been copied into this repository.

| Component | Version verified this round | License | Used for | Original QueryDistill-RL work around it |
| --- | --- | --- | --- | --- |
| PyTorch | 2.13.0+cpu (Windows CPU env); 2.13.0+cu126 (WSL2 CUDA env) | BSD-style | tensor ops, autograd | training/inference orchestration |
| Transformers | 5.15.0 (CPU/GRPO env); 5.6.0 (SFT env, LLaMA-Factory pin) | Apache-2.0 | model loading, tokenizers | backend adapters, unload discipline |
| TRL | 1.10.0 (GRPO env); 0.24.0 (SFT env, LLaMA-Factory pin) | Apache-2.0 | GRPOTrainer (GRPO optimizer) | SQL reward env, reward logging, dataset adapter, callbacks |
| PEFT | 0.20.0 (GRPO env); 0.18.1 (SFT env, LLaMA-Factory pin) | Apache-2.0 | LoRA / QLoRA adapters | target-module detection, merge-then-quantize flow |
| Accelerate | 1.14.0 (GRPO env); 1.11.0 (SFT env, LLaMA-Factory pin) | Apache-2.0 | device management | — |
| bitsandbytes | 0.50.1 | MIT | NF4/4-bit training quantization | config integration, honest version reporting |
| Datasets | 5.0.1 (4.0.0 rejected: broken dill shim on Python 3.14) | Apache-2.0 | HF dataset adapters | schema + split guards |
| sqlglot | 30.17.0 | MIT | SQL AST parsing (safety Layer 1) | full safety policy + decision type |
| pydantic | 2.13.4 | MIT | dataset schema validation | Example/DistillationRecord schemas |
| typer | 0.27.1 | MIT | CLI | all CLI commands |
| pytest | 9.1.1 | MIT | tests | 204-test CPU suite |
| ruff | 0.16.3 | MIT | lint/format | project lint config |
| psutil | 7.2.2 | BSD-3 | hardware doctor | — |
| PyYAML | 6.0.3 | MIT | config files | resolved-config persistence |
| NumPy | 2.4.3 | BSD-3 | bootstrap resampling | paired bootstrap implementation |
| sentencepiece | 0.2.2 | Apache-2.0 | Qwen tokenizer support | — |
| LLaMA-Factory | 0.9.5 (installed in WSL `.venv-sft`; real QLoRA SFT smoke PASS) | Apache-2.0 | QLoRA SFT pipeline | YAML generation/validation, dataset registration, log collection, resume wrapper |
| GPTQModel | 7.3.2 (installed in WSL `.venv-wsl`; real INT4 smoke PASS) | Apache-2.0 | GPTQ INT4 quantization | calibration guard, merge-then-quantize orchestrator, compatibility report |
| vLLM | not installed (serve env incomplete; VLLM_NOT_SMOKE_VERIFIED) | Apache-2.0 | OpenAI-compatible serving | serve config, benchmark client, honest compatibility check |

Environment note: the SFT venv and GRPO/GPTQ venv are separate because
LLaMA-Factory 0.9.5 pins `transformers<=5.6 / accelerate<=1.11 / peft<=0.18.1 /
trl<=0.24`, which conflicts with TRL 1.10. The text-only smoke path uses
documented local torchaudio/torchvision stubs (`scripts/stubs/`) because no
cp314 wheel exists for torch 2.13.0+cu126; audio/vision code paths raise loudly
if ever called.

## What is original in this repository

* `src/querydistill/sql/` - AST safety policy, process-isolated read-only
  executor, db_id allowlist environment, result equivalence verifier.
* `src/querydistill/rewards/` - all reward components and the composite logic.
* `src/querydistill/data/` - dataset schemas, synthetic fixture generator,
  LeakageGuard, data audit.
* `src/querydistill/distillation/` - resumable verify-then-persist pipeline.
* `src/querydistill/training/` - thin wrappers around upstream trainers plus
  the real SQLite GRPO reward wiring and reward logging.
* `src/querydistill/quantization/` - GPTQModel adapter and contamination guard.
* `src/querydistill/serving/` - vLLM config builder + benchmark client.
* `src/querydistill/evaluation/` - unified harness, error buckets, metrics,
  exact McNemar and paired bootstrap.
* All tests and documentation.

## Explicit non-claims

This project does **not** claim to invent GRPO, QLoRA, GPTQ, knowledge
distillation, LLaMA-Factory, TRL or vLLM. It also does not claim DeepSpeed
usage (the deepspeed config directory is a future example and was never run).
