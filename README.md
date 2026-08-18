# QueryDistill-RL

**Efficient Small-LLM Post-Training for Text-to-SQL via Knowledge Distillation,
QLoRA, Verifiable Reinforcement Learning, GPTQ Quantization and vLLM Serving.**

> **Status: PRE-FULL HARDENING (Phase 1.7)**
>
> Real Qwen3-4B Teacher, BIRD execution verification, 88-example paired
> Gold/Distilled QLoRA SFT, real SQLite GRPO learning signal, GRPO
> confirmation, and GPTQ INT4 quantization have all been **pilot-verified**.
> This is **not** production-ready, **not** SOTA, and contains **no formal
> benchmark claims**. The repository is in pre-full hardening: protocol-correct
> GRPO reconfirmation and GPTQ load validation are being recorded; full 6601
> BIRD onboarding is prepared but the full DB set is not yet present locally.
> See `reports/PRE_FULL_HARDENING_REPORT.md`.

---

## Motivation

Text-to-SQL systems usually need a large general LLM to reach good execution
accuracy. This project asks one concrete research question:

> On a single consumer GPU (RTX 4060 Laptop, 8 GB VRAM), how far can a **0.6B
> student model** be pushed with (a) offline teacher distillation, (b) QLoRA
> SFT, and (c) online verifiable RL (GRPO) whose reward is real SQLite
> execution, before being deployed with GPTQ INT4 through vLLM?

The contribution is **not** a new GRPO / QLoRA / GPTQ / distillation algorithm.
The contribution is a complete, safe, auditable integration: data,
verification, safety, reward design, experiment design, evaluation and
quantization benchmarking.

## Pipeline

```
BIRD-like examples (train only for teacher)
        │
        ▼
Teacher (Qwen3-4B, offline, unloaded afterwards)
        │ candidates verified against SQLite gold results
        ▼
Distilled-SFT target ──────────────┐
                                   ▼
Student (Qwen3-0.6B-Base) ── QLoRA SFT (LLaMA-Factory)
        │
        ├── Gold-SFT (target = gold SQL)
        └── Distilled-SFT (target = verified teacher SQL)
                │
                ▼
        TRL GRPO with real SQLite reward environment
                │
                ▼
        LoRA merge → GPTQ INT4 (GPTQModel) → vLLM serving
```

Planned comparison arms (not executed in this round):

1. Student Base
2. Gold-SFT
3. Distilled-SFT
4. Distilled-SFT + GRPO
5. GRPO BF16/FP16 deployment
6. GRPO GPTQ INT4 deployment

Optional: DPO offline baseline (see `docs/DPO_ABLATION_PLAN.md`), self-supervised
CPT. PPO is not implemented in this project.

## Why GRPO (and why not PPO)

* Text-to-SQL has a verifiable database-execution reward; no reward model is
  trained or needed.
* GRPO does not require an independent value model/critic, which is more
  realistic on one consumer GPU.
* Multiple SQL rollouts for the same prompt give a group-relative signal.
* Online rollouts directly exercise the verifier instead of learning from a
  fixed preference dataset.

Standard DPO is offline and needs chosen/rejected pairs, so it is kept only as
an optional ablation (execution-correct = chosen, execution-wrong = rejected).
Full reasoning: `docs/ALGORITHM_DECISIONS.md`.

## LLaMA-Factory role

LLaMA-Factory is used as the mature upstream QLoRA SFT implementation
(`src/querydistill/training/llamafactory_backend.py`). The project generates and
validates its own YAML and dataset registration, invokes the upstream CLI,
collects output paths and trainer logs, saves resolved configs and supports
checkpoint resume. Calling LLaMA-Factory is **not** presented as a contribution.

## SQL RL environment

* `SQLExecutionEnvironment`: model supplies only a `db_id`; the environment
  resolves it through an allowlist registry to a SQLite file. Databases open
  read-only.
* `SafeSQLExecutor` (P0, two layers):
  1. sqlglot AST validation - only one top-level `SELECT` / `WITH ... SELECT` /
     `UNION SELECT`; INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/REPLACE/ATTACH/
     DETACH/VACUUM/write-PRAGMA/TRIGGER/load_extension and multiple statements
     are rejected.
  2. SQLite `mode=ro` + authorizer (SELECT/READ/FUNCTION only, fail-closed),
     progress handler, watchdog `interrupt()`, and **process isolation** so a
     timeout terminates the whole worker.
* `ResultEquivalenceVerifier`: compares execution results (multiset by default,
  ordered when gold has a semantically relevant `ORDER BY`), handles NULL,
  duplicates, integer/float tolerance, strings, empty results, column counts.
  Both-empty results are never accepted unconditionally - see
  `docs/RESULT_EQUIVALENCE.md`.

## Reward design

```
unsafe SQL                    -1.0 (hard)
malformed output              -0.4
parse valid                   +0.05
safety valid                  +0.05
execution success             +0.10
execution equivalent          +1.00 (dominant)
```

A pretty but wrong SQL can earn at most 0.25 before correctness; a correct SQL
earns ~1.25. Empty-empty "correctness" is capped at 0.25 and requires structural
sanity. Full rationale: `docs/REWARD_DESIGN.md`.

## QLoRA / GPTQ / vLLM

* QLoRA is **training-time** memory optimization (4bit NF4, configurable double
  quantization), never final deployment quantization.
* LoRA target modules are detected/validated from the actual Qwen model
  architecture rather than copied from online examples.
* GPTQ INT4 uses GPTQModel; calibration data comes from train/calibration
  splits only (`test contamination guard`).
* vLLM is the serving backend in a separate `serve` environment. GPTQ-kernel
  compatibility is checked at metadata level and verified only by actually
  starting the server - no assumptions.

## Hardware constraints

* Windows 11 + RTX 4060 Laptop 8 GB VRAM + 16 GB RAM; C: ~20 GB free, D: ~189 GB
  free. All large files live on D: (`D:\LLMProjects\QueryDistill-RL`,
  `D:\LLMCache`). See `.env.example`.
* Primary CUDA environment: WSL2 Ubuntu. Native Windows is only used for CPU
  tests in this round.
* This repository never claims production readiness, SOTA, or statistical
  significance from pilot runs. Large formal runs are not started without
  manual approval.

## Current verification status

Real pilot evidence (frozen from earlier phases; see reports):

| Item | Status |
| --- | --- |
| Real Qwen3-4B Teacher | PASS |
| 88-example paired Gold/Distilled SFT | PASS |
| Corrected Distilled-SFT (88 rows, 66 steps) | PASS |
| Canonical `</sql>` stopping | PASS |
| Protocol-correct GRPO reconfirmation | PASS |
| GPTQ INT4 quantization | PASS |
| GPTQ load validation | ENV BLOCKED (CUDA_HOME/nvcc) |
| Full BIRD formal data onboarding | IN PROGRESS (see `artifacts/formal_readiness`) |
| Formal full experiment | NOT RUN |
| vLLM | NOT SMOKE VERIFIED |

Current CPU regression (final commit): `compileall` PASS, `ruff` PASS,
`pytest` = **319 passed, 1 skipped** on fresh clone of the final commit.

No benchmark numbers are claimed. Pilot validation values are engineering
observations only.

## Tests

```bash
python -m compileall src tests
ruff check .
ruff format --check .
pytest -q                      # CPU suite
scripts/gpu_smoke.sh           # GPU smokes (WSL2 + CUDA), marks each PASS/BLOCKED
```

CPU suite covers: config, hardware doctor, output parser, SQL safety, executor
timeout/process isolation, result equivalence, every reward component, reward
hacking attacks, dataset schema, leakage guard, distillation resume, evaluation
metrics, statistics, error buckets, LLaMA-Factory YAML/backend, GPTQ adapter and
vLLM config. GPU tests are marked `@pytest.mark.gpu`.

## Roadmap to full experiment

1. **Pass second code review** (this round; stop here).
2. Download student + teacher, generate the full training-split distillation
   dataset (offline, resumable).
3. Gold-SFT and Distilled-SFT QLoRA runs with matched budgets.
4. GRPO from the Distilled-SFT checkpoint with the real SQLite reward.
5. GPTQ INT4 + vLLM concurrency benchmark (1/4/8).
6. Evaluation harness on the held-out test split + McNemar / paired bootstrap.

## Limitations

* Tiny synthetic fixtures (`tests/fixtures/tiny_sql`) are engineering smoke
  data only; they are never presented as benchmark results.
* The project verifies API paths, data integrity, and pilot capability, not formal accuracy.
* Empty-result equivalence is deliberately conservative; more sophisticated
  sanity checks are future work.
* vLLM/GPTQ kernel behavior on Ada (RTX 4060) is recorded, not assumed.
* No RAG, no multi-agent, no PPO, no reward model, no DeepSpeed claims.

## Attribution

All third-party dependencies, their licenses, and what is original code in this
repository are documented in `THIRD_PARTY.md`, `REFERENCES.md`, `DATA_CARD.md`
and `MODEL_CARD.md`.
