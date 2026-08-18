# First Build Report - QueryDistill-RL

Round: FIRST IMPLEMENTATION / FIRST CODE REVIEW
Generated: 2026-08-17 (UTC+8)

---

## 1. Actual project path

* Windows: `D:\LLMProjects\QueryDistill-RL`
* WSL2: `/mnt/d/LLMProjects/QueryDistill-RL`

## 2. Actual cache paths

* Cache root: `D:\LLMCache` (Windows) = `/mnt/d/LLMCache` (WSL2)
  * `huggingface/` - HF model/tokenizer cache
  * `huggingface/datasets/` - datasets cache
  * `torch/`, `xdg/`, `uv/`, `pip/`, `triton/`, `torch_extensions/`, `wheels/`
* The pre-existing `C:\Users\31465\.cache` is a symlink to `D:\ModelCache` that
  existed before this project started; the project did not create or modify it.

## 3. Is everything on D:?

YES for all project-managed large files.

* Project, data, models, checkpoints, artifacts, runs: `D:\LLMProjects\QueryDistill-RL`
* All HF/torch/pip/uv/triton caches: `D:\LLMCache`
* No application code writes model/cache files to C:.
* Ephemeral-only C: usage: pytest temp dirs and Python's OS temp dir (not models/cache).
* `hardware-doctor` path audit: PROJECT_ROOT on D:, cache root on D: - OK.

## 4. Python environments (real, not assumed)

| Env | Location | Python | Purpose | Key verified versions |
| --- | --- | --- | --- | --- |
| `.venv` | project root (D:) | CPython 3.11.15 (Windows) | CPU tests, CLI, audit, dev | torch 2.13.0+cpu, transformers 5.15.0, trl 1.10.0, peft 0.20.0, accelerate 1.14.0, bitsandbytes 0.50.1, datasets 5.0.1, sqlglot 30.17.0, pydantic 2.13.4, typer 0.27.1, pytest 9.1.1, ruff 0.16.3 |
| `.venv-wsl` | project root (D:) | CPython 3.14.4 (Ubuntu 26.04 WSL2) | GRPO smoke + GPTQ smoke | torch 2.13.0+cu126, transformers 5.15.0, trl 1.10.0, peft 0.20.0, accelerate 1.14.0, bitsandbytes 0.50.1, datasets 5.0.1, sqlglot 30.17.0, gptqmodel 7.3.2 |
| `.venv-sft` | project root (D:) | CPython 3.14.4 (Ubuntu 26.04 WSL2) | LLaMA-Factory QLoRA SFT smoke | torch 2.13.0+cu126, transformers 5.6.0, accelerate 1.11.0, peft 0.18.1, trl 0.24.0, datasets 5.0.1, llamafactory 0.9.5, bitsandbytes 0.50.1 |
| `.venv-serve` | project root (D:) | CPython 3.14.4 | vLLM serve (INCOMPLETE) | not smoke verified; see section 15/21 |

Environment split is deliberate:

* LLaMA-Factory 0.9.5 pins `transformers<=5.6, accelerate<=1.11, peft<=0.18.1,
  trl<=0.24`, which conflicts with TRL 1.10, so SFT and GRPO cannot share one env.
* vLLM was intended to live in a separate serve env per the project spec.

## 5. CUDA / PyTorch

* Host: Windows 11, Intel i7-14650HX, 16 GB RAM (WSL sees ~5.78 GB default).
* GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8187 MiB VRAM.
* Driver: 566.24 (Windows nvidia-smi reports CUDA 12.7; WSL nvidia-smi 565.75).
* torch (WSL): 2.13.0+cu126, `torch.cuda.is_available() == True`, bf16 True.
* torch (Windows): 2.13.0+cpu, CUDA not available by design (CPU test env).
* WSL hardware-doctor: CUDA availability True, bf16 True, GPU visible.

## 6. Model downloads

* Student `Qwen/Qwen3-0.6B-Base`: DOWNLOADED to
  `models/qwen3-0.6b-base` (1.2 GB) via `hf-mirror.com` because
  `huggingface.co` is unreachable from this network. Xet backend was disabled
  (`HF_HUB_DISABLE_XET=1`) after the mirror's CAS endpoint returned 401.
* Teacher `Qwen/Qwen3-4B`: NOT downloaded (first-round rule F obeyed).
* No BIRD dataset downloaded. No other model downloaded.

## 7. File tree (abbreviated)

```
QueryDistill-RL/
├── pyproject.toml  README.md  LICENSE  .env.example  .env(gitignored)
├── THIRD_PARTY.md  REFERENCES.md  DATA_CARD.md  MODEL_CARD.md  CHANGELOG.md
├── configs/{sft,grpo,quant,serving,deepspeed}/
├── requirements/{core,train,serve,dev}.txt
├── src/querydistill/
│   ├── cli.py  config.py  hardware.py  utils.py
│   ├── data/{schema,fixtures,leakage,audit,dataset}.py
│   ├── sql/{safety,executor,environment,verifier}.py
│   ├── outputs/{parser,prompting}.py
│   ├── rewards/{base,format,parse,safety,execution,correctness,composite}.py
│   ├── distillation/{backends,pipeline}.py
│   ├── training/{llamafactory_backend,grpo_backend,callbacks}.py
│   ├── quantization/gptq.py
│   ├── evaluation/{harness,metrics,errors,statistics}.py
│   ├── serving/vllm.py
│   └── reporting/report.py
├── tests/ (15+ test modules, 204 tests)
├── scripts/ (gpu_smoke.sh, serve_vllm.sh, benchmark_vllm.py,
│             run_inference_smoke.py, run_cpu_checks.sh, package_release.py,
│             stubs/{torchaudio,torchvision}_stub.py)
├── docs/ (ALGORITHM_DECISIONS, RESULT_EQUIVALENCE, REWARD_DESIGN, SAFETY,
│          EXPERIMENT_PLAN, DPO_ABLATION_PLAN, SCALING, FIRST_BUILD_REPORT)
├── artifacts/smoke/{inference,sft,grpo,gptq,vllm}/
├── reports/  runs/  checkpoints/  models/  data/
└── .github/workflows/ci.yml
```

## 8. SQL safety implementation

Two layers, all original code (`src/querydistill/sql/`):

1. `safety.py`: sqlglot AST allowlist. Exactly one top-level SELECT /
   WITH-SELECT / UNION-SELECT. Rejects INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/
   REPLACE/ATTACH/DETACH/VACUUM/write-PRAGMA/TRIGGER/load_extension and multiple
   statements (semicolon tricks). Trailing semicolon and comments do not count
   as extra statements.
2. `executor.py`: process-isolated SQLite with `mode=ro`, `PRAGMA query_only`,
   authorizer allowlisting only SELECT/READ/FUNCTION/RECURSIVE (denies
   `load_extension`), progress handler + watchdog `interrupt()` + hard process
   terminate. `max_rows` and `max_execution_ms` enforced.

Covered by tests: DROP, DELETE, UPDATE, INSERT, ATTACH, DETACH,
`PRAGMA writable_schema`, VACUUM, CREATE, multiple statements, semicolon trick,
comments, WITH-SELECT, nested SELECT, UNION, recursive CTE timeout, expensive
Cartesian product, duplicate SQL tags.

## 9. Result verifier implementation

`ResultEquivalenceVerifier` compares execution results, not SQL strings:

* multiset row comparison when gold has no ORDER BY; ordered comparison when it does
* NULL vs `"NULL"` distinct, duplicates significant, integer/float tolerance
  1e-6, strings exact, column count must match
* both-empty is NEVER accepted unconditionally: requires FROM + candidate
  tables subset of gold tables + known-schema sanity, and is capped as
  `empty_structural` (documented in docs/RESULT_EQUIVALENCE.md)

## 10. Reward implementation

Original components (`src/querydistill/rewards/`):

```
unsafe SQL         -1.0 hard
malformed output   -0.4
parse valid        +0.05
safety valid       +0.05
execution success  +0.10
execution equiv    +1.00 (empty-structural capped at +0.25)
```

Every rollout logs a full trace to `reward_samples.jsonl` (parse, safety,
execution, verification). Reward hacking suite covers SELECT 1 / SELECT NULL /
constant / always-empty / schema-independent / malformed / hidden statements /
comment bypass / expensive CTE / timeout / duplicate tags.

## 11. LeakageGuard implementation

`src/querydistill/data/leakage.py` enforces: split ID overlap, exact and
normalized question overlap, gold SQL / gold result never in policy prompt,
file-declared split mismatch, teacher candidates from test split. All guarded
paths are asserted in `tests/test_leakage_guard.py`.

## 12. LLaMA-Factory integration - REAL SMOKE PASS

* Env: `.venv-sft` (LLaMA-Factory 0.9.5 + its pinned transformer stack).
* Command: `querydistill train-sft --config configs/sft/gold_smoke.yaml`.
* Result: **PASS**. 2 optimizer steps on the tiny fixture; trainer log shows
  `loss 2.488 -> 2.439`; adapter saved at `checkpoints/sft/gold_smoke/`
  (`adapter_model.safetensors`, `checkpoint-2`, `trainer_log.jsonl`).
* The backend generated the LLaMA-Factory YAML, registered the project dataset
  (`artifacts/sft/gold_smoke/dataset/{train.json,dataset_info.json}`), invoked
  the real upstream CLI, collected the trainer log, and saved resolved config.
* Real integration issues fixed and recorded:
  * LLaMA-Factory 0.9.5 uses `cutoff_len`, not `max_seq_length`.
  * `resume_from_checkpoint: true` on an empty output dir is invalid; backend
    now disables it when no checkpoint exists.
  * LLaMA-Factory pins `datasets<=4.0` whose dill shim crashes on Python 3.14;
    `.venv-sft` uses datasets 5.0.1 + `DISABLE_VERSION_CHECK=1` (documented).
  * Text-only smoke uses a documented torchaudio stub because no cp314
    torchaudio wheel matches torch cu126 (audio paths raise loudly).

## 13. TRL GRPO integration - REAL SMOKE PASS

* Env: `.venv-wsl` (TRL 1.10.0, transformers 5.15.0).
* Command: `querydistill train-grpo --config configs/grpo/smoke.yaml`
  (num_generations=2, 2 steps, 8 tiny examples, max_completion_length=96).
* Result: **PASS**. 2 real optimizer steps; trainer metrics logged; adapter
  saved to `artifacts/smoke/grpo/adapter`; 2 reward samples written.
* Reward is real: `reward_samples.jsonl` traces show the base student produced
  malformed output (no `<sql>` tags), the pipeline scored
  `format=-0.4, parse=-0.4, total=-0.8`, and execution was correctly skipped.
  No random/hard-coded/fake reward anywhere.
* Artifacts: `environment.json`, `resolved_config.yaml`,
  `reward_samples.jsonl`, `trainer_log`, `metrics.json`, `README.md`.

## 14. GPTQModel integration - REAL SMOKE PASS

* Env: `.venv-wsl` (GPTQModel 7.3.2, torch cu126).
* Command: `querydistill quantize-gptq --config configs/quant/gptq_int4_smoke.yaml`.
* Result: **PASS**. GPTQ INT4, group_size=128, sym=True, desc_act=False.
* Calibration: 8 examples from **train split only**
  (`calibration_manifest.json` records ids/splits; contamination guard active).
* Size: pre-quantized 1136.91 MB -> quantized 516.27 MB (-54.59%).
* Quantized checkpoint: `artifacts/smoke/gptq/quantized/`.
* Text-only smoke uses a documented torchvision stub (GPTQModel imports
  torchvision only for its InternVL model class; Qwen quantization never calls it).

## 15. vLLM integration - VLLM_NOT_SMOKE_VERIFIED

* Code complete and unit tested: `VLLMServeConfig`, `build_server_command`,
  OpenAI-compatible benchmark client (TTFT/P50/P95/tokens/reqs/peak VRAM),
  `check_compatibility`, `scripts/serve_vllm.sh`, `scripts/benchmark_vllm.py`.
* Status: **VLLM_NOT_SMOKE_VERIFIED** (`artifacts/smoke/vllm/status.json`).
* Real reasons:
  1. vLLM does not run on native Windows; WSL2 is required.
  2. In WSL2 the driver (CUDA 12.7) forces torch 2.13.0+cu126; vLLM 0.27.1
     needs a multi-GB CUDA package set.
  3. WSL `/tmp` is a 2.9 GB tmpfs and pip unpacking repeatedly failed with
     `ENOSPC`; the dedicated `.venv-serve` could not be completed within the
     first-round budget.
* No server was started and no PASS is claimed. This is NOT a blocker for
  first code review (per project rule Y).

## 16. pytest raw summary (final run)

```
python -m pytest -q
203 passed, 1 skipped in 126.81s
```

The skip is `tests/test_hardware.py::test_gpu_probe_sees_cuda_when_present`,
correctly skipped because the Windows CPU test env has no CUDA torch. GPU tests
are marked `@pytest.mark.gpu`.

## 17. ruff raw summary (final run)

```
ruff check .            -> All checks passed!
ruff format --check .   -> 70 files already formatted (pass)
ruff version: 0.16.3
```

## 18. compileall status

```
python -m compileall -q src tests -> exit 0 (pass)
```

## 19. GPU smoke status (only real runs)

| Smoke | Env | Status | Evidence |
| --- | --- | --- | --- |
| Student base inference | `.venv-wsl` | PASS | real Qwen3-0.6B generation, protocol parsed, SQLite reward trace; artifacts/smoke/inference/status.json |
| LLaMA-Factory QLoRA SFT | `.venv-sft` | PASS | 2 steps, loss 2.488→2.439, adapter + checkpoint saved |
| TRL GRPO + real SQLite reward | `.venv-wsl` | PASS | 2 steps, 2 logged real reward traces, adapter saved |
| GPTQModel INT4 | `.venv-wsl` | PASS | 1.11GB→0.50GB, train-only calibration manifest |
| vLLM serve | - | VLLM_NOT_SMOKE_VERIFIED | status.json documents real blockers |

## 20. Peak VRAM

Only measured values are reported. No peak-VRAM instrumentation was attached
to the training smokes, so **no peak VRAM number is written here** (per policy:
never invent measurements). Total GPU VRAM reported by the probes: 8187 MiB.

## 21. Current blockers

* vLLM live smoke: not verified (see section 15). Not a code-review blocker.
* `huggingface.co` unreachable from this network; `hf-mirror.com` is used and
  documented. This must be revisited before downloading the 4B teacher.
* Python 3.14 ecosystem friction in WSL (Ubuntu 26.04 default):
  datasets 4.0/dill pickle bug, LLaMA-Factory version pins, no cp314
  torchaudio/torchvision for torch cu126 (documented stubs only for text path).
* 9p filesystem makes WSL site-package I/O slow (imports can take minutes when
  cold). Training itself ran at expected speed once loaded.

## 22. What was NOT run in this round

* Teacher Qwen3-4B download / real teacher generation (rule F).
* BIRD or any large dataset download.
* Formal SFT / GRPO / GPTQ / vLLM benchmarks.
* Distilled-SFT real smoke (only mock distillation records + Gold-SFT smoke).
* DPO, DeepSpeed, PPO, reward model, RAG, multi-agent.
* vLLM concurrency sweep.
* Statistical claims on tiny fixture data.

## 23. Third-party attribution

Recorded in `THIRD_PARTY.md` and `REFERENCES.md`:

* Official APIs only: PyTorch, Transformers, TRL, PEFT, Accelerate,
  bitsandbytes, datasets, sqlglot, pydantic, typer, pytest, ruff, psutil,
  PyYAML, NumPy, sentencepiece.
* LLaMA-Factory 0.9.5 (Apache-2.0): QLoRA SFT pipeline, used via generated
  YAML + CLI; explicitly not presented as project contribution.
* GPTQModel 7.3.2 (Apache-2.0): GPTQ INT4 quantization adapter.
* vLLM: not installed; config/benchmark code only.
* All SQL safety, reward, verification, leakage, distillation, evaluation and
  integration code in `src/querydistill/` is original to this repository.

## 24. Known limitations

* Tiny synthetic fixtures are engineering smoke data, never benchmark results.
* Base student produced malformed protocol output in GRPO smoke (expected for
  an untrained base model; no accuracy claim).
* Empty-result equivalence is deliberately conservative.
* GPTQ smoke quantized the BASE student, not a trained adapter (allowed by
  rule X for round 1).
* vLLM kernel behavior on Ada remains unverified.
* DeepSpeed config is a future example only; no DeepSpeed claim.

---

PROJECT STATUS:
READY_FOR_FIRST_CODE_REVIEW
