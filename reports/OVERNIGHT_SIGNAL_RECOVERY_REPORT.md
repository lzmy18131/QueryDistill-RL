# Overnight Signal Recovery Report — Phase 1.6

> Status: FINAL — GPTQ_CAPABILITY_RETENTION_FAILED (all GRPO/SFT gates passed; GPTQ load blocked by missing Marlin/Ninja).

## Timeline

| Stage | Started | Ended | Duration | Status |
|---|---|---|---|---|
| Preflight | 2026-08-18T01:43 | 2026-08-18T01:43 | - | PASS |
| Corrected Distilled-SFT | 2026-08-18T01:44 | 2026-08-18T01:57 | 168.8s | PASS (66 steps) |
| StopAfterSqlClose + tests | 2026-08-18T01:43 | 2026-08-18T02:20 | - | PASS |
| CPU regression (pytest) | 2026-08-18T02:20 | 2026-08-18T02:22 | 138s | 308 passed, 1 skipped |
| Validation Base/Gold/Distilled | 2026-08-18T03:33 | 2026-08-18T04:00 | ~27 min | PASS |
| Generation diagnostic | 2026-08-18T04:01 | 2026-08-18T04:12 | ~11 min | PASS |
| GRPO signal | 2026-08-18T04:19 | 2026-08-18T04:31 | ~12 min | PASS |
| GRPO confirmation | 2026-08-18T04:32 | 2026-08-18T04:40 | ~8 min | PASS |
| GPTQ | 2026-08-18T04:49 | 2026-08-18T05:06 | ~17 min | QUANT PASS / LOAD FAIL |

## 1. Preflight

- diagnostic verified = 24
- collection verified = 64
- union verified = 88
- all_candidates verified = 88
- paired_manifest = 88
- Missing IDs = 0
- Extra IDs = 0

Artifacts: `artifacts/overnight/preflight.json`, `artifacts/overnight/paired/full_verified_targets.jsonl`, `artifacts/overnight/paired/equality_report.json`.

## 2. 88 vs 64 P0

- Root cause: `signal_distilled_sft.yaml` used `teacher_collection/candidates.jsonl` (64 verified) instead of `teacher_collection/all_candidates.jsonl` (88 verified union).
- Before: Gold rows 88, Distilled rows 64, optimizer steps Gold 66 / Distilled 48.
- After: Distilled rows 88, optimizer steps 66.
- Old 64-example artifacts archived under `archived/distilled_sft_64_bug/`.

## 3. Corrected Distilled-SFT

- Rows: 88
- Optimizer steps: 66
- Epochs: 3.0
- Train loss: 0.5837
- Runtime: 168.8s
- Adapter SHA256: `fb860e915b441f836e8b8ea986ff240dc344c2e6c9f88ef1a137043c02280a9b`
- Peak VRAM: not captured for this run (no monitor started); see metrics for honest null.
- Artifacts: `artifacts/overnight/distilled_sft/`.

## 4. SQL termination

Implemented `StopAfterSqlClose` in `src/querydistill/generation/stopping.py`.

- Checks generated region only (prompt is excluded).
- Stateful per-sequence finished mask.
- Used by validation and generation diagnostic tools.
- Tests: `tests/test_stop_after_sql_close.py` (7 passed).

## 5. Validation

New stop-at-`</sql>` validation on 32 validation_tuning examples (deterministic + sampled, 64 records per model).

| Model | format_valid | parse_valid | safe | execution_success | strict_EX | stop_at_close | truncation | mean_tokens |
|---|---|---|---|---|---|---|---|---|
| Base | 17.19% | 6.25% | 14.06% | 7.81% | 0% | 7.81% | 92.19% | 140.91 |
| Gold-SFT | 95.31% | 95.31% | 93.75% | 32.81% | 7.81% | 95.31% | 4.69% | 61.45 |
| Corrected Distilled-SFT | 96.88% | 96.88% | 93.75% | 50.00% | 14.06% | 96.88% | 3.12% | 61.02 |

Corrected Distilled-SFT passes the SFT readiness gate: format >= 60%, parse >= 50%, and execution_success_count > 0.

## 6. SFT extension

Not needed — Corrected Distilled-SFT passed the readiness gate on the first run.

## 7. Generation diagnostic

24 validation_tuning prompts x 4 completions = 96 raw completions.

| Metric | Value |
|---|---|
| format_valid | 91.67% |
| parse_valid | 91.67% |
| safe | 90.62% |
| execution_success | 38.54% |
| strict_correct | 20.83% |
| raw completion unique | 96.88% |
| normalized SQL unique | 91.67% |
| nonzero reward-std groups | 18 / 24 (75%) |
| stop_at_sql_close | 20.83% |
| truncation | 79.17% |
| multiple SQL blocks | 1.04% |
| format-invalid but SQL extractable | 25% |

Diversity gate PASS (>=3 groups and >=20% fraction).

## 8. Sampling probe

Not needed — generation diagnostic diversity gate passed (18/24 nonzero reward-std groups).

## 9. GRPO signal run

- Config: `configs/experiment/overnight_grpo_signal.yaml`
- Run ID: `grpo-signal`
- Steps: 8, num_generations: 2, per_device_train_batch_size: 2 (set to satisfy TRL 1.10 `generation_batch_size % num_generations`)
- Status: `GRPO_INTEGRATION_PASS`
- Learning signal: `GRPO_LEARNING_SIGNAL_PASS`
- Nonzero reward-std groups: 3 (steps 2/4/6)
- Nonzero grad-norm steps: 3 (steps 2/4/6)
- parameter_delta_l2: 0.04681784
- trainable_param_sha changed: true
- changed_parameter_tensor_count: 392
- reward_samples_logged: 16
- real SQLite reward: true

## 10. GRPO confirmation

- Config: `configs/experiment/overnight_grpo_confirmation.yaml`
- Run ID: `grpo-confirmation`
- Steps: 20, num_generations: 2, per_device_train_batch_size: 2
- Status: `GRPO_INTEGRATION_PASS`
- Learning signal: `GRPO_LEARNING_SIGNAL_PASS`
- Nonzero reward-std groups: 6
- Nonzero grad-norm steps: 6
- parameter_delta_l2: 0.06094999
- trainable_param_sha changed: true
- changed_parameter_tensor_count: 392
- reward_samples_logged: 40
- real SQLite reward: true

GRPO_CONFIRMATION_PASS.

## 11. Validation comparison

Same 32 validation_tuning examples (64 records incl. sampled).

| Model | format | parse | safe | exec_success | strict_EX |
|---|---|---|---|---|---|
| Corrected Distilled-SFT | 96.88% | 96.88% | 93.75% | 50.00% | 14.06% |
| GRPO Confirmation | 98.44% | 98.44% | 98.44% | 50.00% | 18.75% |

No severe protocol regression (no drop > 20 percentage points).

## 12. GPTQ

- Input: GRPO Confirmation adapter
- Config: `configs/experiment/overnight_gptq.yaml`
- Quantization: GPTQModel INT4, group_size=128, sym=True, desc_act=False
- Calibration: train_core, 16 samples (train-only)
- Merged checkpoint: `checkpoints/signal_recovery/grpo_confirmation_merged`
- Quantized output: `checkpoints/signal_recovery/grpo_confirmation_gptq/quantized` (528 MB)
- Status: `PASS`

Quantized model validation was attempted but **failed to load** due missing Marlin kernels in the WSL environment (`ModuleNotFoundError: Marlin torch.ops kernels are not properly installed` / Ninja missing). Quantization itself succeeded (status PASS, 528 MB INT4 checkpoint saved). Per phase rules this is recorded as `GPTQ_CAPABILITY_RETENTION_FAILED` and no retraining was performed.

## 13. External blockers

None so far.

## 14. Final status

PROJECT STATUS: `GPTQ_CAPABILITY_RETENTION_FAILED`

All earlier gates passed:
- Paired data: 88 / 88 exact IDs
- Corrected Distilled-SFT: 66 steps, format 96.88%, parse 96.88%, exec_success > 0
- Generation diagnostic: 24x4, 18/24 nonzero reward-std groups
- GRPO signal: `GRPO_LEARNING_SIGNAL_PASS`
- GRPO confirmation: `GRPO_CONFIRMATION_PASS`
- GPTQ quantization: PASS (INT4, 528 MB)

The only blocker is GPTQ **load/validation** in the current WSL environment: Marlin kernels are not installed (missing Ninja), so the quantized checkpoint could not be loaded for validation. This is recorded honestly as `GPTQ_CAPABILITY_RETENTION_FAILED`; no model was retrained and no result was faked.


## 4b. SQL executor fix (WSL/CUDA spawn timeout)

Discovered that `SafeSQLExecutor` using `spawn` timed out after CUDA initialization in WSL. Switched to `fork` on POSIX (worker never touches CUDA), making real SQLite execution reliable and fast in CUDA processes. Old invalid validation outputs archived under `artifacts/overnight/validation_old_spawn_timeout/`.
