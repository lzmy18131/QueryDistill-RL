# Data Card

## Current datasets

### querydistill-tiny-synthetic v1.0.0

Engineering smoke fixture.

| Property | Value |
| --- | --- |
| Purpose | Engineering smoke data for pipeline verification ONLY |
| Status | FIRST IMPLEMENTATION ROUND fixture |
| Source | Hand-authored, deterministic synthetic data (no personal/private data) |
| Databases | shop, school, company (SQLite, 4 tables each) |
| Examples | 42 (train 21 / dev 12 / test 9) |
| Coverage | select, filter, join, aggregation, group by, order, limit, subquery, CTE, empty result |
| Generation | `querydistill make-fixtures` via `src/querydistill/data/fixtures.py` |
| Files | `tests/fixtures/tiny_sql/databases/*.db`, `tests/fixtures/tiny_sql/db_registry.json`, `data/tiny_sql/examples.jsonl` |
| Licensing | Original synthetic content, MIT (same as repository) |
| Benchmark use | FORBIDDEN - never present tiny-fixture numbers as benchmark results |

## Split policy

* `train`: SFT/GRPO training and teacher candidate generation.
* `dev`: model selection only.
* `test`: final evaluation only. Gold SQL and gold results of test examples
  never enter a model prompt (enforced by LeakageGuard + tests).
* `calibration`: GPTQ calibration (train/calibration only; dev/test banned).
* `validation_tuning`: engineering validation; excluded from training.
* `formal_validation`: held-out formal evaluation; evaluation allowed, every
  training/teacher/GPTQ-calibration path is forbidden.

## Real BIRD formal data (Phase 1.9)

- Full filtered train source: 6601 rows.
- DB Registry: 80 required DBs resolved (69 train + 11 eval), 0 missing,
  0 hash conflicts, 80/80 quick_check, 80/80 schema hashes.
- Formal split (frozen, stable SHA-256 IDs):
  - `formal_train.jsonl`: 6225 rows (`split=train`)
  - `engineering_validation.jsonl`: 120 rows (`split=validation_tuning`)
  - `formal_validation.jsonl`: 256 rows (`split=formal_validation`)
- Union = 6601, pairwise overlap = 0.
- Mini-Dev final SELECT-500: 500; 20 historically exposed; 480 unexposed.
- Gold execution audit completed with `audit_timeout_ms=30000`; slow Gold SQL
  is preserved as runtime evidence, not deleted.

Full SQLite database files and the 36MB `formal_train.jsonl` are not committed
to Git; they are regenerated locally by `tools/build_final_pretraining_gate.py`.

## Distillation records

Every distillation record stores: example_id, teacher_model,
teacher_model_revision, teacher_prompt_version, candidate_index, candidate_sql,
parse_valid, safe, execution_success, execution_equivalent, generation_config,
created_at. Candidates are verified against SQLite gold execution before being
eligible as Distilled-SFT targets.
