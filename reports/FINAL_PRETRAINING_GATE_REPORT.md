# Final Pretraining Gate Report

Phase 1.9 — FINAL EXECUTION SEMANTICS, FORMAL DATA MATERIALIZATION & PRE-TRAINING GO/NO-GO GATE

## 1. Executive Result

**Project Status: FORMAL_EXPERIMENT_READY** (if all gates pass; otherwise the
specific blocker is listed in Section 16).

* Full BIRD filtered train source: 6601 verified rows.
* DB Registry: 80 required DBs resolved, 0 missing, 0 hash conflicts.
* Gold execution: full audit completed with explicit `audit_timeout_ms=30000`.
* Formal split: 6225 train / 120 engineering validation / 256 formal validation.
* Stable BIRD IDs: 6601/6601 unique, cross-process stable.
* Formal model training: **NOT YET RUN** (waits for human-provided FORMAL BIRD
  EXPERIMENT MATRIX).

## 2. Airline Root Cause

- Old DB hash: `11e57cdf...` (169,099,264 bytes; Postgres-flavoured airline,
  missing `Airports`/`Airlines`/`Air Carriers`).
- Canonical DB hash: `50308a01...` (122,699,776 bytes; official BIRD train
  airline database).
- Evidence: 66/66 airline Gold SQL executed in a direct read-only smoke against
  the canonical DB; old DB produced `no such table` for all 66.
- Action: old DB moved to `data/bird/quarantine/`; canonical DB installed in
  the registry.
- 66-example rerun: 66/66 success, 0 schema mismatch, 0 timeout in the final
  30s audit.

## 3. Safety REPLACE Root Cause

- sqlglot AST: `SELECT REPLACE(...)` is `exp.Replace` inside a `Select`;
  `REPLACE INTO ...` is `exp.Command`; `INSERT OR REPLACE ...` is `exp.Insert`.
- Fix: removed `replace` from the blanket forbidden-key set; scalar `Replace`
  remains allowed inside SELECT, while `Command`/`Insert`/top-level non-SELECT
  still fail closed.
- 40 historical REPLACE false positives: 40/40 success, 0 timeout,
  0 unexpected in the final 30s audit.

## 4. Timeout Semantics

- Historical audit incorrectly reused the candidate/RL default `3000 ms`.
- This round separates:
  - `audit_timeout_ms = 30000` for Gold correctness audit;
  - `candidate_execution_timeout_ms = 3000` (unchanged, not frozen) for
    Teacher/GRPO candidate execution.
- Full 6601 Gold audit was run at 30s.  Historical timeout buckets were derived
  from the richer single run: recovered ≤10s, recovered 10–30s, residual 30s.
- Optional 120s diagnostic: run on the 3 residual 30s timeouts; 2 recovered
  as slow-but-executable, 1 remained persistent.

## 5. Gold Final Audit

| Category | Count |
| --- | --- |
| Source rows | 6601 |
| Historical success under 3s | 5848 |
| Fixed airline success | 66/66 |
| Fixed REPLACE success | 40/40 |
| Actual success under 3s | 6525 |
| Success 3–10s | 64 |
| Success 10–30s | 9 |
| Slow success 30–120s | 2 |
| Normal success under 30s | 6598 |
| Persistent timeout | 1 |
| SQLite errors | 0 |
| Unsafe failures | 0 |
| Other/unknown | 0 |
| Executable/classified | 6601/6601 |

## 6. Gold Latency Profile

P50 / P90 / P95 / P99 / max = 258.01 / 426.81 / 934.54 / 3529.59 /
25069.57 ms.  See `artifacts/final_pretraining_gate/gold_latency_profile.json`.

## 7. Candidate Timeout Compatibility

See `artifacts/final_pretraining_gate/execution_policy_compatibility.json`.
Gold queries exceeding 3s / 5s / 10s = 76 / 54 / 12.  Formal GRPO timeout is
not decided here; evidence only.

## 8. Formal Split Materialization

- `data/bird/splits/formal_train.jsonl`: 6225 rows, split=`train`.
- `data/bird/splits/engineering_validation.jsonl`: 120 rows (alias
  `validation_tuning.jsonl` kept in sync).
- `data/bird/splits/formal_validation.jsonl`: 256 rows, split=`formal_validation`.
- Union = 6601, pairwise overlap = 0.

## 9. Stable ID Migration

- Canonical ID = SHA-256 of canonical JSON identity
  (`db_id`, raw question, evidence, gold SQL), prefixed `bird-train-`.
- Mini-Dev uses official `question_id` as `bird-dev-{question_id}`.
- No Python built-in `hash()` is used for formal BIRD identity.
- Migration manifest: `artifacts/final_pretraining_gate/formal_id_migration.jsonl`.

## 10. Leakage

- ID overlap with Mini-Dev: 0.
- Normalized question overlap: 0.
- Question+evidence hash overlap: 0.
- Question+gold SQL hash overlap: 0.
- Exact duplicate question warnings in train source: 1 (recorded, not deleted).
- Normalized duplicate question groups in train source: 2 (recorded, not deleted).
- Mini-Dev exposed: 20/500.

## 11. DB Registry Final Lock

- Resolved: 80/80.
- Missing: 0.
- Hash conflicts: 0.
- Quick check: 80/80.
- Schema hashes: 80/80.
- Airline canonical hash: `50308a01...`.

## 12. Formal Protocol Lock

- All hash fields are real lowercase 64-hex SHA-256 values.
- `db_registry_hash` is valid, not `[]`/null.
- See `artifacts/final_pretraining_gate/formal_protocol_lock.json`.

## 13. CPU Regression

- `compileall`: PASS
- `ruff check`: PASS
- `ruff format --check`: PASS
- `pytest`: 346 passed, 1 skipped (local; fresh-clone numbers below)

## 14. Fresh Clone Exact Commit

- Final commit SHA: `d87d48b682b8b75dbf6561d37f8bc58764da8f26`
- Tested commit SHA: `d87d48b682b8b75dbf6561d37f8bc58764da8f26`
- Fresh clone import/compileall/ruff: PASS
- Fresh clone pytest: 344 passed, 3 skipped
- Tracked pyc: 0; secret hits: 0

## 15. Remaining Limitations

- GPTQ load validation remains environment-blocked (WSL lacks CUDA_HOME/nvcc).
- Formal GRPO/Teacher/SFT/GPTQ execution policy not frozen; requires Formal
  Experiment Matrix.
- Slow Gold SQL queries are preserved as runtime evidence, not deleted.

## 16. Final Status

**PROJECT STATUS: FORMAL_EXPERIMENT_READY**

**EXECUTION POLICY: CANDIDATE_TIMEOUT_REQUIRES_FORMAL_MATRIX_DECISION**

STOP. No formal training is started in this round.
