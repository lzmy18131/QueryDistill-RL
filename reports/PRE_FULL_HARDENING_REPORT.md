# Pre-Full Hardening Report — Phase 1.7

Status: IN PROGRESS.

## 1. Repository repair

- Fixed `.gitignore` so `/data/` does not ignore `src/querydistill/data/`.
- Added `src/querydistill/data/**/*.py` to Git.
- Added small reproducible data fixtures (`data/tiny_sql`, `data/bird/raw`, `data/bird/splits`, `data/bird/examples`, `data/bird/db_registry.json`).
- Tracked pyc / __pycache__ count: 0.
- Secret scan: 0 hits.

## 2. Fresh clone reproducibility

TBD.

## 3. SQL stopping alignment

- Canonical `StopAfterSqlClose` + `SQL_CLOSE_TAG` in `src/querydistill/generation/stopping.py`.
- GRPO `generation_kwargs={"stop_strings": ["</sql>"]}` via `build_grpo_generation_kwargs()`.
- Protocol verification result: see generation_protocol/metrics.json.

## 4. Diagnostic reward fix

- `tools/run_generation_diagnostic.py` now always calls `CompositeReward.score_once`.
- Records `reward_internal_error` instead of None->0.
- Tests added.

## 5. Strict GRPO gate

- Added `_reward_signal_stats`, `strict_grpo_signal_gate_pass`, `strong_confirmation_gate_pass`, semantic variance group detection.
- Tests added.

## 6. Protocol-correct GRPO reconfirm

- Config: `configs/experiment/pre_full_grpo_protocol.yaml`
- Run ID: `grpo-protocol-reconfirm`
- Steps: 12, num_generations: 2, per_device_train_batch_size: 2
- SQL-close stopping via `SqlStoppingGRPOTrainer` (passes `StopAfterSqlClose` stopping criteria)
- Status: `GRPO_INTEGRATION_PASS`
- Learning signal: `GRPO_LEARNING_SIGNAL_PASS`
- strict_grpo_signal_gate_pass: true
- strong_confirmation_gate_pass: true
- groups: 12
- nonzero reward-std groups: 7
- semantic variance groups: 7
- nonzero grad steps: 7
- parse-valid completions: 22
- execution successes: 7
- parameter_delta_l2: 0.04820805
- changed LoRA tensors: 392
- all rewards finite: true
- Mean completion length during GRPO: 38-100 (vs old 128 clipped), clipped_ratio 0.5

## 7. Validation observation

Same 32 validation_tuning examples (64 records incl. sampled).

| Model | format | parse | safe | exec_success | strict_EX | stop_at_close | truncation | multiple_sql |
|---|---|---|---|---|---|---|---|---|
| Corrected Distilled-SFT | 96.88% | 96.88% | 93.75% | 50.00% | 14.06% | 96.88% | 3.12% | 0% |
| Protocol-Correct GRPO | 100% | 100% | 98.44% | 50.00% | 15.62% | 100% | 0% | 0% |

Observed pilot values only; no statistical significance claimed.

## 8. GPTQ load validation

TBD.

## 9. Full BIRD onboarding

- 6601 filtered train rows confirmed.
- formal_train_core = 6481 (6601 minus 120 validation_tuning).
- DB registry missing 60 DBs -> FULL_BIRD_DATABASE_INCOMPLETE.

## 10. Leakage audit

- formal_train_core vs validation_tuning overlap = 0.
- formal_train_core vs Mini-Dev overlap = 0.
- validation_tuning vs Mini-Dev overlap = 0.
- duplicate normalized question count = 1.

## 11. Documentation update

- README / MODEL_CARD / DATA_CARD updated.

## 12. Remaining blockers

- Full BIRD DB set missing (60 DBs).
- GPTQ load validation pending.

## 13. CPU regression

TBD.

## 14. Final status

TBD.
