# Data Card

## Current dataset: querydistill-tiny-synthetic v1.0.0

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

## Planned real dataset (later round)

BIRD (or another clearly documented public Text-to-SQL benchmark) will be
converted into this schema after the first code review. Until then the
repository contains no downloaded BIRD data and makes no BIRD claims.

## Distillation records

Every distillation record stores: example_id, teacher_model,
teacher_model_revision, teacher_prompt_version, candidate_index, candidate_sql,
parse_valid, safe, execution_success, execution_equivalent, generation_config,
created_at. Candidates are verified against SQLite gold execution before being
eligible as Distilled-SFT targets.
