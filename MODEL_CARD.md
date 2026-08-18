# Model Card

## Round-2 model usage

| Role | Model | Round-2 status |
| --- | --- | --- |
| Student (the training target) | Qwen/Qwen3-0.6B-Base | Downloaded locally at `models/qwen3-0.6b-base`; inference/SFT/GRPO/GPTQ smokes use it |
| Teacher (offline candidate generation only) | Qwen/Qwen3-4B | Download FORBIDDEN in Round 2; only mock-teacher and config/dry-run are used |

## Intended use

Research/educational small-LLM post-training pipeline for Text-to-SQL on a
single consumer GPU. Not intended for production database systems.

## Training data

Round 2: synthetic tiny fixtures only (`tests/fixtures/tiny_sql`). All
training/calibration paths are train-only by construction; dev/test are
excluded by `SplitPolicy`. Later rounds: public benchmark train split with the
leakage rules in `src/querydistill/data/leakage.py`.

## Deployment plans (not fully executed in Round 2)

* BF16/FP16 merged LoRA checkpoint
* GPTQ INT4 checkpoint via GPTQModel (calibration restricted to
  train/calibration splits)
* vLLM OpenAI-compatible serving with kernel compatibility recorded, not
  assumed; Round 2 records `VLLM_NOT_SMOKE_VERIFIED` when the serve environment
  is not ready.

## Ethics / safety

Generated SQL is treated as untrusted code: sqlglot AST allowlist plus SQLite
read-only mode, authorizer, progress handler and process isolation. The model
never receives database filesystem paths, only allowlisted db_ids. Real model
contexts never contain gold SQL/results (`RealModelContext`).

## Known limitations

* 0.6B student accuracy is expected to be far below large models.
* Empty-result correctness is deliberately conservative (partial credit only).
* No deployment claim is made until evaluation and review are complete.
* Round 2 focuses on correctness/chain integrity; it is not a benchmark run.
