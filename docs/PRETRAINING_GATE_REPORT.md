# QueryDistill-RL Round 2.1 + Round 2.2 — Final Pre-Training Gate Report

Round name: **PRE-TRAINING GATE FIX / FINAL PRE-TRAINING FIX**

This report covers the formal-config and experiment-integrity fixes from Round
2.1 and the final Round 2.2 release-consistency fixes. No BIRD, no Teacher 4B,
no formal training, no new algorithm, and no vLLM environment work were
performed.

## Project status

```
PROJECT STATUS: READY_FOR_FULL_EXPERIMENT
```

All P0 items PASS. P1/P2 engineering items are also covered by tests and code.

## CPU acceptance

Executed on native Windows CPU venv and saved to
`artifacts/gates/pretraining_cpu_checks.txt`:

```text
python -m compileall src tests scripts   -> PASS
ruff check .                             -> PASS
ruff format --check .                    -> PASS
pytest -q                                -> 281 passed, 1 skipped
```

The single skip is the CUDA-availability hardware probe in a CPU-only
environment.

## P0 items

### P0-1: Formal GRPO local config must match smoke chain — PASS

- Source: `configs/grpo/local.yaml`, `src/querydistill/cli.py`, `src/querydistill/utils.py`.
- Tests:
  - `tests/test_round2_1.py::test_formal_grpo_config_parses`
  - `tests/test_round2_1.py::test_formal_grpo_config_requires_sft_artifact`
  - `tests/test_round2_1.py::test_formal_protocol_is_sql_only`
- Verification:
  - `configs/grpo/local.yaml` now uses `base_model_path`, `init_adapter_path`, `init_merged_model_path`, `require_plan: false`.
  - `model_id` is no longer accepted; strict config loader rejects unknown fields.

### P0-2: Formal GPTQ / vLLM artifact chain — PASS

- Source: `configs/quant/gptq_int4_local.yaml`, `configs/serving/vllm_gptq_local.yaml`, `configs/serving/vllm_smoke.yaml`.
- Tests: `tests/test_round2_1.py::test_formal_config_artifact_chain`.
- Verification:
  - GPTQ local consumes `checkpoints/grpo/distilled_grpo_local/adapter`.
  - GPTQ outputs `checkpoints/merged/distilled_grpo_local` and `checkpoints/gptq/distilled_grpo_int4`.
  - vLLM local consumes `checkpoints/gptq/distilled_grpo_int4`.
  - vLLM smoke consumes the actual smoke GPTQ output `artifacts/smoke/gptq/quantized`.

### P0-3: Real Teacher provenance must be real — PASS

- Source: `src/querydistill/distillation/backends.py::TeacherConfig`, `src/querydistill/cli.py`, `src/querydistill/distillation/pipeline.py`.
- Tests:
  - `tests/test_round2_1.py::test_real_teacher_provenance_not_mock`
  - `tests/test_round2_1.py::test_real_teacher_generation_config_in_fingerprint`
  - `tests/test_round2_1.py::test_teacher_model_change_changes_fingerprint`
  - `tests/test_round2_1.py::test_teacher_sampling_change_changes_fingerprint`
- Verification:
  - `distill generate --backend transformers` now builds a real `TeacherConfig` and records `teacher_model`, `teacher_revision`, `prompt_version`, and sampling/generation fields.
  - Mock backend remains explicitly `mock-teacher-1.0`; real teacher is never recorded as mock.

### P0-4: Teacher / SFT / GRPO output protocol unified SQL-only — PASS

- Source: `src/querydistill/distillation/pipeline.py` (`require_plan=False` default), existing SFT/GRPO defaults.
- Tests: `tests/test_round2_1.py::test_formal_protocol_is_sql_only`.
- Verification:
  - Distillation default `require_plan=False`.
  - GRPO default `require_plan=False`.
  - SFT default `include_plan=False`.
  - Prompt protocol for `include_plan=False` is `<sql>...</sql>` only.

### P0-5: Paired Gold-SFT vs Distilled-SFT — PASS

- Source: `src/querydistill/data/paired.py`, `src/querydistill/cli.py`, `configs/sft/*_local.yaml`.
- Tests:
  - `tests/test_round2_1.py::test_paired_manifest_controls_gold_subset`
  - `tests/test_round2_1.py::test_gold_and_distilled_use_same_paired_manifest`
- Verification:
  - `QLoRAConfig` now supports `paired_manifest_path`.
  - Distilled-SFT writes the paired manifest; Gold-SFT reads the same manifest and filters to identical `example_ids`.

### P0-6: verified_teacher_coverage math — PASS

- Source: `src/querydistill/data/paired.py`.
- Tests: `tests/test_round2_1.py::test_verified_teacher_coverage_math`.
- Verification:
  - 3 requested train examples, 1 verified teacher target -> `requested_count=3`, `paired_count=1`, `verified_teacher_coverage=1/3`.

### P0-7: Training split cannot be expanded by config — PASS

- Source: `src/querydistill/data/split_policy.py`, `src/querydistill/cli.py`, `src/querydistill/training/grpo_backend.py`, `src/querydistill/distillation/pipeline.py`, `src/querydistill/quantization/gptq.py`.
- Tests:
  - `tests/test_round2_1.py::test_sft_config_cannot_enable_test`
  - `tests/test_round2_1.py::test_grpo_config_cannot_enable_test`
  - `tests/test_round2_1.py::test_distillation_config_cannot_enable_test`
  - `tests/test_round2_1.py::test_gptq_calibration_cannot_enable_test`
- Verification:
  - `assert_training_splits` enforces `{"train"}` at SFT/GRPO/Distillation boundaries.
  - `assert_calibration_splits` enforces `{"train","calibration"}` at GPTQ calibration boundary.

### P0-8: audit-data overall ok includes distillation leakage — PASS

- Source: `src/querydistill/data/audit.py`.
- Tests: `tests/test_round2_1.py::test_audit_fails_when_distillation_contains_test_data`.
- Verification:
  - When distillation records are supplied, `reports["ok"]` now also requires `distillation.leakage.clean == True`.

## P1 items

### P1-1: Distillation crash-resume — PASS

- Source: `src/querydistill/distillation/pipeline.py`.
- Tests: `tests/test_round2_1.py::test_distillation_resume_after_real_midrun_crash`.
- Verification:
  - RUNNING manifest is written before the first teacher call.
  - A simulated backend crash after one example leaves `status=RUNNING`; a new pipeline with `--resume` verifies fingerprint, skips completed candidates, and completes the rest.

### P1-2: Distillation gold executes once per example — PASS

- Source: `src/querydistill/distillation/pipeline.py`.
- Tests: `tests/test_round2_1.py::test_distillation_gold_executes_once_per_example`.
- Verification:
  - With `num_candidates=4`, gold SQL executes exactly once for the example; candidates reuse the cached gold result.

### P1-3: GoldResultCache database fingerprint — PASS

- Source: `src/querydistill/rewards/composite.py`, `src/querydistill/training/grpo_backend.py`.
- Tests: `tests/test_round2_1.py::test_gold_cache_invalidates_when_database_changes`.
- Verification:
  - Cache path now includes a database fingerprint; changing the DB fingerprint makes old cache entries miss.

### P1-4: GRPO reward traces bounded/cleared — PASS

- Source: `src/querydistill/training/grpo_backend.py`, `src/querydistill/training/callbacks.py`.
- Tests: `tests/test_round2_1.py::test_reward_trace_buffer_is_bounded_or_cleared`.
- Verification:
  - `SQLRewardFunction` keeps only the latest batch.
  - `RewardSampleLogger` clears the trace buffer after persisting the batch.

### P1-5: Run artifact isolation — PASS

- Source: `src/querydistill/training/grpo_backend.py`, `src/querydistill/cli.py`.
- Tests:
  - `tests/test_round2_1.py::test_new_run_refuses_nonempty_output`
  - `tests/test_round2_1.py::test_resume_requires_matching_run_identity`
- Verification:
  - Non-empty output dirs are refused unless `--resume`.
  - `run_identity.json` stores config/dataset/base/init hashes; resume verifies them.

### P1-6: vLLM benchmark api-key end-to-end — PASS

- Source: `src/querydistill/cli.py`, `scripts/benchmark_vllm.py`, `src/querydistill/serving/vllm.py`.
- Tests: `tests/test_round2_1.py::test_cli_benchmark_propagates_api_key`.
- Verification:
  - CLI and script accept `--api-key` and fall back to `VLLM_API_KEY`, passing it to `benchmark(..., api_key=...)`.

### P1-7: Inference smoke real latency + README order — PASS

- Source: `scripts/run_inference_smoke.py`.
- Tests: `tests/test_round2_1.py::test_inference_readme_matches_current_status`.
- Verification:
  - Real `backend.generate()` is timed as `generation_latency_ms`.
  - Status is written before README; README receives the payload directly and never reads a stale status file.

### P1-8: GRPO steps use trainer global_step — PASS

- Source: `src/querydistill/training/grpo_backend.py`.
- Tests: `tests/test_round2_1.py::test_grpo_steps_use_global_step`.
- Verification:
  - `_learning_signal(..., global_step=2)` reports `steps=2` instead of `len(log_history)`.

### P1-9: GPU smoke orchestrator mandatory failure summary — PASS

- Source: `scripts/gpu_smoke.sh`.
- Tests: `tests/test_round2_1.py::test_gpu_smoke_script_has_mandatory_failure_summary`.
- Verification:
  - Mandatory stages increment `MANDATORY_FAILURES`; final script writes `artifacts/smoke/overall_status.json` and exits non-zero if any mandatory stage failed.

### P1-10: Evaluation ModelSpec / loader explicit — PASS

- Source: `src/querydistill/evaluation/modelspec.py`, `src/querydistill/cli.py`.
- Tests: `tests/test_round2_1.py::test_model_spec_identity`.
- Verification:
  - `ModelSpec` captures stage/base/adapter/merged/quantized/quantization/manifest/hash.
  - `evaluate --output` writes `model_identity.json` with backend/model/split/type/quantization metadata.

## P2 items

### P2-1: Scripts cleanup — PASS

- Source: `scripts/`.
- Tests: `tests/test_round2_1.py::test_scripts_directory_is_clean`.
- Verification:
  - Only formal scripts remain:
    `benchmark_vllm.py`, `gpu_smoke.sh`, `package_release.py`, `run_cpu_checks.sh`, `run_inference_smoke.py`, `serve_vllm.sh`.
  - Temporary `_*.sh`/stub scripts are gone; environment pitfalls are in `docs/TROUBLESHOOTING.md`.

### P2-2: Formal config static validation — PASS

- Source: `src/querydistill/utils.py::strict_dataclass_from_dict`, `src/querydistill/cli.py`.
- Tests: `tests/test_round2_1.py::test_strict_config_loader_rejects_unknown_fields`.
- Verification:
  - CLI no longer silently filters unknown YAML keys. Unknown fields such as `model_id` in GRPO config raise `UnknownConfigFieldError`.

### P2-3: GRPO / Distillation artifact logging — PASS

- Source: `src/querydistill/training/grpo_backend.py`, `src/querydistill/distillation/pipeline.py`.
- Verification:
  - GRPO status/artifacts include `run_id`.
  - Distillation manifests/results include `run_id` and `run_fingerprint` (dataset/teacher/generation).
  - Existing honest GRPO status remains `GRPO_INTEGRATION_PASS` / `GRPO_LEARNING_SIGNAL_INSUFFICIENT`.

## Round 2.2 final fixes

### P0-1: Delivery tree and pytest consistency — PASS

- Source: `scripts/`, `tests/test_round2_1.py::test_scripts_directory_is_clean`,
  `tests/test_round2_2.py::test_scripts_directory_exact_set_again`.
- Verification:
  - `scripts/` contains exactly the 6 formal entry points.
  - CPU checks were re-run on the same tree and saved to `artifacts/gates/pretraining_cpu_checks.txt`.
  - `tools/check_release_consistency.py` verifies ZIP hashes, scripts set, extracted compileall, and structural tests; final run result: **OK**.

### P0-2: Formal Distilled-SFT supports non-100% Teacher coverage — PASS

- Source: `configs/sft/distilled_local.yaml`, `src/querydistill/data/paired.py`, `src/querydistill/cli.py`.
- Tests:
  - `tests/test_round2_2.py::test_formal_distilled_local_has_strict_false`
  - `tests/test_round2_2.py::test_distilled_local_dataset_build_coverage_two_thirds`
- Verification:
  - `strict_distilled: false` in the formal local config.
  - 3 requested, 2 verified → `paired_count=2`, `requested_count=3`, `coverage=2/3`.
  - No gold fallback; the paired subset is used by both Gold and Distilled arms.

### P0-3: GRPO --resume truly usable — PASS

- Source: `src/querydistill/training/grpo_backend.py`, `src/querydistill/cli.py`.
- Tests:
  - `tests/test_round2_2.py::test_grpo_resume_flag_does_not_change_training_fingerprint`
  - `tests/test_round2_2.py::test_grpo_resume_preserves_original_run_id`
  - `tests/test_round2_2.py::test_grpo_resume_passes_latest_checkpoint_to_trainer`
  - `tests/test_round2_2.py::test_grpo_resume_rejects_changed_model_dataset_or_init_artifact`
- Verification:
  - `_training_config_payload` excludes `resume`, `dry_run`, `run_id`.
  - Resume inherits the original `run_id`.
  - `_resume_checkpoint()` finds the latest `checkpoint-*`; no checkpoint raises instead of silently restarting.

### P0-4: Formal Evaluation Loader — PASS

- Source: `src/querydistill/evaluation/modelspec.py::load_model`, `src/querydistill/cli.py`.
- Tests:
  - `tests/test_round2_2.py::test_model_loader_base`
  - `tests/test_round2_2.py::test_model_loader_adapter_uses_peft`
  - `tests/test_round2_2.py::test_model_loader_merged`
  - `tests/test_round2_2.py::test_model_loader_gptq`
- Verification:
  - `load_model` branches by `ModelSpec.stage`: base AutoModel, adapter Base+PeftModel, merged AutoModel, GPTQ quantized AutoModel.
  - `evaluate --backend transformers` now accepts `--model-spec`, or `--stage` + `--base-model/--adapter/--model-path`, and writes `model_identity.json` from the spec.

### P1-1: Distillation resume no redundant Teacher calls — PASS

- Source: `src/querydistill/distillation/pipeline.py`.
- Tests:
  - `tests/test_round2_2.py::test_distillation_resume_does_not_call_teacher_for_completed_examples`
  - `tests/test_round2_2.py::test_distillation_resume_preserves_run_id`
- Verification:
  - Fully completed examples are skipped before `backend.generate()`.
  - Resume preserves the manifest `run_id`.

### P1-2: Compact reward logging default — PASS

- Source: `src/querydistill/training/callbacks.py`, `src/querydistill/training/grpo_backend.py`.
- Tests:
  - `tests/test_round2_2.py::test_default_reward_log_is_compact`
  - `tests/test_round2_2.py::test_debug_reward_log_can_include_full_trace`
- Verification:
  - Default records are compact: run_id, component rewards, parse/safety/execution/verification flags, error type, row counts.
  - Full prompt/completion/trace only when `debug_full_trace=true`.

### P1-3: GRPO QLoRA config/docs unified — PASS

- Source: `configs/grpo/local.yaml`.
- Verification:
  - Formal GRPO local now uses `quantize_4bit: true`, matching the QLoRA claim in `docs/ALGORITHM_DECISIONS.md`.

### P2-1: Stale smoke artifacts annotation — PASS

- Verification:
  - README/report explicitly state that existing GPU artifacts are **historical smoke** from before Round 2.2 logic.
  - New `generation_latency_ms`, `global_step`, and `resume` behavior are verified by CPU regression tests, not by old artifacts.

## Algorithm wording constraint

The project continues to describe Distillation as **Verified Sequence-Level /
Response Distillation**. No logit-level KD, KL loss, or teacher-logit claims are
made in README/docs/code.

## Historical GPU smoke note

The existing `artifacts/smoke/*/status.json` files were produced before the
Round 2.2 changes. They are kept as historical evidence of the earlier smoke
chain and are **not** used to claim Round 2.2 `generation_latency_ms`,
`global_step`, or `resume` behavior. Those behaviors are covered by the CPU
regression tests listed above.

## Deliverable

```text
QueryDistill-RL-final-pretraining-review.zip
```

Included: `src`, `tests`, `configs`, `docs`, small fixtures, small smoke
artifacts, reports, `artifacts/gates/pretraining_cpu_checks.txt`. Excluded:
`.venv*`, `models`, large checkpoints, GPTQ weights, HF/pip caches, full
datasets.
