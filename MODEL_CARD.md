# Model Card

## Model usage

| Role | Model | Status |
| --- | --- | --- |
| Student (training target) | Qwen/Qwen3-0.6B-Base | Local `models/qwen3-0.6b-base`; real SFT/GRPO/GPTQ pilots use it |
| Teacher (offline candidate generation) | Qwen/Qwen3-4B | Real 4-bit offline inference, `enable_thinking=False`; unloaded after generation |

## Intended use

Research/educational small-LLM post-training pipeline for Text-to-SQL on a
single consumer GPU. Not intended for production database systems.

## Training data

Real BIRD filtered train has been used for pilot experiments (88 paired
examples for SFT/GRPO). A `validation_tuning` split (120 IDs) is reserved and
never enters training. Formal 6601-row train onboarding is prepared; full DB
set is not yet complete locally. All training/calibration paths are train-only
by construction; dev/test are excluded by `SplitPolicy` and leakage rules in
`src/querydistill/data/leakage.py`.

## Deployment plans (not fully executed)

* BF16/FP16 merged LoRA checkpoint
* GPTQ INT4 checkpoint via GPTQModel (calibration restricted to
  train/calibration splits)
* vLLM OpenAI-compatible serving with kernel compatibility recorded, not
  assumed; the project records `VLLM_NOT_SMOKE_VERIFIED` when the serve
  environment is not ready.

## Ethics / safety

Generated SQL is treated as untrusted code: sqlglot AST allowlist plus SQLite
read-only mode, authorizer, progress handler and process isolation. The model
never receives database filesystem paths, only allowlisted db_ids. Real model
contexts never contain gold SQL/results (`RealModelContext`).

## Known limitations

* 0.6B student accuracy is expected to be far below large models.
* Empty-result correctness is deliberately conservative (partial credit only).
* No deployment claim is made until evaluation and review are complete.
* The project focuses on correctness/chain integrity and pilot evidence; it is not a benchmark run.
