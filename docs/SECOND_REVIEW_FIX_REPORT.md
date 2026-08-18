# QueryDistill-RL Round 2 — Second Review Fix Report

Round name: **Correctness, Data Integrity & Experiment-Chain Hardening**

This report records the Round 2 code-review fixes, regression tests, CPU
acceptance results, and the honest GPU smoke status. No BIRD, no Teacher 4B,
no formal training, no benchmark numbers, and no fabricated PASS are included.

## Project status

```
PROJECT STATUS: READY_FOR_SECOND_CODE_REVIEW
```

P0 correctness fixes are complete and covered by tests. GPU smoke chain is
re-run with the Round 2 corrected chain; vLLM remains optional and is honestly
recorded as `VLLM_NOT_SMOKE_VERIFIED` if not available.

## CPU acceptance

Executed on native Windows with the project CPU venv:

```text
python -m compileall src tests scripts   -> PASS
ruff check .                             -> PASS
ruff format --check .                    -> PASS
pytest -q                                -> 237 passed, 1 skipped
```

The single skip is the CUDA-availability hardware probe in a CPU-only
environment.

## P0 fixes

### P0-1: All training paths fail-closed train-only — PASS

- Source: `src/querydistill/data/split_policy.py`, `src/querydistill/distillation/pipeline.py`,
  `src/querydistill/training/grpo_backend.py`, `src/querydistill/quantization/gptq.py`,
  `src/querydistill/cli.py`.
- Tests:
  - `tests/test_round2_hardening.py::test_split_policy_train_only_excludes_dev_test`
  - `tests/test_round2_hardening.py::test_calibration_split_policy_excludes_dev_test`
  - `tests/test_round2_hardening.py::test_distillation_dry_run_split_report_is_train_only`
  - `tests/test_round2_hardening.py::test_grpo_dry_run_split_report_is_train_only`
  - `tests/test_evaluation_metrics.py::test_evaluation_requires_explicit_split`
- Verification: dry-run distillation/GRPO/GPTQ report excluded dev/test and
  evaluation CLI refuses a missing `--split`.

### P0-2: DistillationRecord stores parsed SQL — PASS

- Source: `src/querydistill/data/schema.py`, `src/querydistill/distillation/pipeline.py`.
- Tests:
  - `tests/test_dataset_schema.py::test_distillation_record_schema_fields`
  - `tests/test_round2_hardening.py::test_distillation_record_stores_parsed_sql_not_raw_output`
  - `tests/test_round2_hardening.py::test_distilled_target_not_nested`
  - `tests/test_round2_hardening.py::test_distilled_target_single_sql_block`
- Verification: `data/distilled/mock_smoke.jsonl` was regenerated with
  `candidate_sql` containing only the parsed SQL and `raw_candidate_output`
  retaining the teacher output.

### P0-3: Distilled-SFT never falls back to gold SQL — PASS

- Source: `src/querydistill/data/paired.py`, `src/querydistill/data/dataset.py`,
  `src/querydistill/cli.py`.
- Tests:
  - `tests/test_round2_hardening.py::test_distilled_missing_target_fails`
  - `tests/test_round2_hardening.py::test_paired_gold_distilled_same_ids`
  - `tests/test_round2_hardening.py::test_verified_candidate_selection_deterministic`
- Verification: missing teacher targets raise `DistilledTargetMissingError`;
  paired arms use identical example ids; candidate selection is deterministic
  by `min_candidate_index`.

### P0-4: GRPO initializes from Distilled-SFT — PASS (config/dry-run + real smoke)

- Source: `src/querydistill/training/grpo_backend.py`.
- Tests:
  - `tests/test_grpo_backend.py::test_grpo_config_requires_sft_initialization`
  - `tests/test_round2_hardening.py::test_grpo_config_honored_in_dry_run`
- Verification: `GRPOSmokeConfig` refuses base-only initialization, and the
  Round 2 `configs/grpo/smoke.yaml` points `init_adapter_path` to the
  Distilled-SFT artifact. Real GPU smoke writes `initialization_manifest.json`
  when training runs (see GPU smoke section).

### P0-5: Stage artifact chain Base→SFT→GRPO→Merge→GPTQ→vLLM — PASS

- Source: `src/querydistill/artifacts/manifest.py`, `src/querydistill/quantization/gptq.py`,
  `src/querydistill/training/grpo_backend.py`.
- Tests:
  - `tests/test_round2_hardening.py::test_stage_artifact_chain_manifests`
- Verification: every stage writes `artifact_manifest.json` with stage,
  input/output artifact, base model, adapter, config hash. GPTQ smoke config now
  consumes the GRPO adapter (`configs/quant/gptq_int4_smoke.yaml`).

### P0-6: ResultEquivalenceVerifier strictness — PASS

- Source: `src/querydistill/sql/verifier.py`, `src/querydistill/sql/executor.py`.
- Tests:
  - `tests/test_result_equivalence.py::test_truncated_candidate_never_strict_equivalent`
  - `tests/test_result_equivalence.py::test_truncated_gold_never_strict_equivalent`
  - `tests/test_result_equivalence.py::test_both_empty_requires_structural_sanity`
  - `tests/test_result_equivalence.py::test_alias_difference_can_be_execution_equivalent`
  - `tests/test_result_equivalence.py::test_column_count_mismatch_fails`
  - `tests/test_evaluation_metrics.py::test_partial_equivalence_not_accuracy`
- Verification: truncated results are never strict; both-empty results are
  partial credit only; alias differences do not block execution equivalence;
  column count mismatches still fail.

### P0-7: Gold data removed from real model contexts — PASS

- Source: `src/querydistill/outputs/context.py`, `src/querydistill/distillation/backends.py`,
  `src/querydistill/evaluation/harness.py`, `scripts/run_inference_smoke.py`.
- Tests:
  - `tests/test_round2_hardening.py::test_real_model_context_has_no_gold`
  - `tests/test_round2_hardening.py::test_teacher_backend_cannot_receive_gold`
  - `tests/test_round2_hardening.py::test_evaluation_backend_cannot_receive_gold`
- Verification: `RealModelContext.safe_keys()` is exactly
  `{example_id, db_id, split}`; real teacher/evaluation backends raise if a
  caller passes `gold_sql`; inference smoke now generates exactly once with a
  safe context.

## P1 fixes

### P1-1 Teacher backend 4-bit + provenance — PASS (config/dry-run only)

- Source: `src/querydistill/distillation/backends.py`.
- Tests: `tests/test_distillation_resume.py::test_mock_backend_contract` and
  provenance field inspection in the backend implementation.
- No Teacher 4B download was performed.

### P1-2 Distillation resume fingerprint — PASS

- Source: `src/querydistill/distillation/pipeline.py`.
- Tests:
  - `tests/test_round2_hardening.py::test_distillation_resume_rejects_model_change`
  - `tests/test_round2_hardening.py::test_distillation_resume_accepts_same_fingerprint`

### P1-3 Reward / SQL execution single calculation — PASS

- Source: `src/querydistill/rewards/composite.py`.
- Tests:
  - `tests/test_round2_hardening.py::test_candidate_executes_once_per_reward`
  - `tests/test_round2_hardening.py::test_gold_result_cache_roundtrip`
  - `tests/test_round2_hardening.py::test_gold_cache_invalidates_on_dataset_change`

### P1-4 GRPO learning-signal smoke — GRPO_INTEGRATION_PASS / GRPO_LEARNING_SIGNAL_INSUFFICIENT

No fabricated `GRPO_LEARNING_SIGNAL_PASS` is claimed. The Round 2 GPU smoke
completed two GRPO steps from the Distilled-SFT adapter with real SQLite
rewards, but `reward_std=0` and `grad_norm=0`; the status is therefore
`GRPO_LEARNING_SIGNAL_INSUFFICIENT` (honest). The initialization manifest
records `initialization_source=sft_adapter`, adapter sha256, and trainable
parameter fingerprints before/after.

### P1-5 GRPO example mapping by metadata — PASS

- Source: `src/querydistill/data/dataset.py`, `src/querydistill/training/grpo_backend.py`.
- Tests: `tests/test_round2_hardening.py::test_duplicate_prompt_examples_do_not_overwrite`.

### P1-6 GRPO config honored — PASS

- Source: `src/querydistill/training/grpo_backend.py`, `src/querydistill/data/dataset.py`.
- Tests:
  - `tests/test_round2_hardening.py::test_grpo_config_honored_in_dry_run`
  - `tests/test_round2_hardening.py::test_prompt_budget_enforced`

### P1-7 No fake generic plan — PASS

- Source: `src/querydistill/data/dataset.py::plan_from_sql`.
- Tests: `tests/test_round2_hardening.py::test_sft_has_no_placeholder_plan`.

### P1-8 Evaluation split + metrics — PASS

- Source: `src/querydistill/data/split_policy.py`, `src/querydistill/evaluation/metrics.py`,
  `src/querydistill/evaluation/harness.py`.
- Tests: `tests/test_evaluation_metrics.py` (split required, parse-valid,
  partial-not-accuracy).

### P1-9 Evaluation model loaders / model identity — PASS

- Source: `src/querydistill/cli.py` writes `model_identity.json` next to any
  `evaluate --output` report, recording backend, model path, split, model type,
  and quantization metadata when available.
- Tests: `tests/test_cli_smoke.py::test_evaluate_mock_gold_writes_model_identity`.
- Evaluation backends support base/merged/GPTQ paths through
  `TransformersModelBackend`; artifact manifests record stage identity.

### P1-10 vLLM bearer auth — PASS

- Source: `src/querydistill/serving/vllm.py`.
- Tests: `tests/test_round2_hardening.py::test_vllm_client_sends_bearer_token`.

### P1-11 GPU smoke orchestrator — PASS (script)

- Source: `scripts/gpu_smoke.sh` now uses `.venv-sft` / `.venv-wsl` and the
  corrected chain (inference → Gold-SFT → Distilled-SFT → GRPO-from-SFT →
  GPTQ-from-GRPO → optional vLLM).

## P2 fixes

### P2-1 Inference artifact consistency / single generation — PASS

- Source: `scripts/run_inference_smoke.py` now generates once and reuses the
  same completion for evaluation and reward; it uses a test split and safe
  context (no `gold_sql`).

### P2-2 Documentation consistency — PASS

- `README.md`, `MODEL_CARD.md`, and this report reflect Round 2 state and the
  real smoke results; no stale first-round numbers are kept.

### P2-3 .gitignore — PASS

- `.venv*/` entries cover `.venv-wsl`, `.venv-sft`, `.venv-serve`.

### P2-4 Script cleanup — PASS

- Removed temporary `_*.sh` patch scripts and `scripts/stubs`; formal scripts
  remain (`gpu_smoke.sh`, `run_cpu_checks.sh`, `run_inference_smoke.py`,
  `serve_vllm.sh`, `benchmark_vllm.py`, `package_release.py`).
- Environment pitfalls are recorded in `docs/TROUBLESHOOTING.md`.

## GPU smoke summary

Statuses below are the real status files under `artifacts/smoke/*/status.json`
after the Round 2 corrected chain was attempted.

| Stage | Status |
| --- | --- |
| Student base inference | PASS |
| Gold-SFT QLoRA smoke | PASS |
| Distilled-SFT QLoRA smoke | PASS |
| GRPO from Distilled-SFT | `GRPO_INTEGRATION_PASS` / `GRPO_LEARNING_SIGNAL_INSUFFICIENT` |
| GPTQ INT4 from GRPO | PASS (consumed `checkpoints/merged/distilled_grpo_local`) |
| vLLM serve | VLLM_NOT_SMOKE_VERIFIED / optional |

Detailed evidence (resolved configs, split reports, initialization manifests,
reward logs, metrics) is in the corresponding `artifacts/smoke/*/` directories.

## Deliverable

`QueryDistill-RL-second-review.zip` is produced by:

```bash
python scripts/package_release.py
```

It includes `src`, `tests`, `configs`, `docs`, small fixtures, small smoke
artifacts, resolved configs, metrics, logs; excludes `.venv*`, `models/`,
large checkpoints, HF/pip caches and complete datasets.
