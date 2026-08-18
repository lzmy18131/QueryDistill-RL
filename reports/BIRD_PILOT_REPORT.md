# BIRD Pilot Report

Project: **QueryDistill-RL — Experiment Phase 0**

Date: 2026-08-17

## Status

```text
PROJECT STATUS: BIRD_PILOT_BLOCKED
```

The BIRD onboarding and protocol-lock engineering is complete and verified, but
the real Qwen3-4B Teacher and real GPU SFT/GRPO pilot were **not** run in this
session. The pipeline is ready; the remaining blockers are real-Teacher
download/execution and a real tiny GRPO step.

## What passed

### Data onboarding and audit
- `querydistill prepare-bird` created:
  - `data/bird/examples/train.jsonl` — 50 examples
  - `data/bird/examples/dev.jsonl` — 20 examples
  - `data/bird/db_registry.json`
- Real BIRD source used:
  - `birdsql/bird23-train-filtered`
  - `birdsql/bird_mini_dev`
  - SQLite DBs from `birdsql/bird-critic-1.0-sqlite` (mini-dev) and `birdsql/six-gym-sqlite` (pilot train)
- `querydistill audit-data`:
  - train: **ok=True**, gold execution 50/50
  - dev: **ok=True**, gold execution 20/20
- `docs/BIRD_DATA_REPORT.md` generated:
  - Chat-serialized prompt token P50=822, P90=846, P95=858, P99=874, max=896
  - Truncation rate at max_prompt_length:
    - 512: 71.43%
    - 768: 62.86%
    - 1024: 0%
    - 1536: 0%
  - **Frozen `max_prompt_length=1024`** in `configs/experiment/bird_grpo.yaml`.

### Prompt/template protocol lock
- `student_prompt_version = bird-v1`
- `student_chat_template = qwen_chatml`
- SFT rows and GRPO/Eval share the same raw user content.
- Chat template applied in:
  - `build_prompt_rows` (GRPO)
  - `TransformersModelBackend.generate` (Evaluation)
  - `TransformersTeacherBackend.generate` (Teacher, with `enable_thinking=False`)
- Regression tests:
  - `tests/test_bird_onboarding.py::test_sft_and_grpo_use_same_user_content`
  - `tests/test_bird_onboarding.py::test_chat_serialization_is_stable`
  - `tests/test_bird_onboarding.py::test_teacher_backend_uses_chat_template`

### Official evaluator
- Official scripts downloaded to `third_party/bird_eval/` (not claimed as original).
- `src/querydistill/evaluation/bird_eval.py` provides:
  - `export_predictions`
  - `internal_execution_accuracy`
  - `run_official_evaluator`
- Official EX evaluator run on the 20-example Mini-Dev pilot subset with gold SQL as predictions:
  - **Official EX = 100.00%** (simple 100, moderate 100, challenging 100)

### Experiment configs
- `configs/experiment/` created:
  - `bird_base.yaml`
  - `bird_gold_sft.yaml`
  - `bird_distilled_sft.yaml`
  - `bird_grpo.yaml`
  - `bird_gptq.yaml`
- All no longer point to `data/tiny_sql` or `tests/fixtures`.

### Mock pipeline dry-runs
- Mock Teacher distillation: 5 verified candidates written to `data/bird/distilled/teacher.jsonl`.
- Distilled-SFT dry-run: paired manifest `requested_count=50`, `paired_count=5`, `coverage=0.1`.
- Gold-SFT dry-run: uses the same paired manifest and produced exactly 5 rows.
- GRPO dry-run: 20 examples, config validated.
- GPTQ dry-run: config validated.

## What remains blocked

1. **Real Teacher Qwen3-4B not downloaded/executed.**
   - Current teacher evidence is from `mock-teacher-1.0`, not a real Qwen3-4B.
   - Required: run real teacher on 5–20 BIRD train examples, record `teacher_model`, `teacher_revision`, sampling config, valid SQL rate, and verified coverage.
2. **Real tiny SFT not executed.**
   - Only dry-run dataset build was performed.
3. **Real tiny GRPO not executed.**
   - Only dry-run config was performed.
   - Required: verify real reward variance and non-zero grad/update evidence.
4. **Real GPTQ from GRPO not executed.**
   - Only dry-run config was performed.

## Next actions (after this review)

1. Download/allow Qwen3-4B Teacher to D: cache.
2. Run `distill generate --backend transformers --teacher-config configs/teacher/...` on 5–20 BIRD train examples.
3. Run Distilled-SFT smoke (real 2 steps).
4. Run GRPO smoke from Distilled-SFT (real 2 steps).
5. Run GPTQ smoke from GRPO output.
6. Run official evaluator on Mini-Dev predictions.
7. Re-run audit + report; if all pass, flip status to `READY_FOR_FULL_BIRD_EXPERIMENT`.
