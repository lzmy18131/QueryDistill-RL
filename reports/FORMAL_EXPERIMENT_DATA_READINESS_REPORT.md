# Formal Experiment Data Readiness Report — Phase 1.8

Status: FINAL — FULL_BIRD_GOLD_EXECUTION_AUDIT_FAILED

## 1. Executive Status

TBD after gold execution audit.

## 2. Source Provenance

See `artifacts/formal_readiness/source_provenance.json`, `mini_dev_source_provenance.json`.

## 3. 6601 Train Verification

- Filtered train rows: 6601
- Unique IDs: 6601
- Unique train DBs: 69

## 4. Full DB Registry

- Required train DBs: 69
- Required eval DBs: 11
- Union DBs: 80
- Missing DBs: 0
- Hash conflict: airline (existing vs official)
- Read-only / quick_check / schema introspection: PASS for all resolved DBs
- See `artifacts/formal_readiness/db_registry_manifest.json`

## 5. Gold Structural Audit

- Total: 6601
- Parse success: 6601
- SELECT/WITH SELECT compatible: 6601
- See `artifacts/formal_readiness/gold_structural_audit.json` (from pre_full_hardening, re-verified).

## 6. Gold Execution Audit

- Total: 6601
- Execution success: 5848
- Failures: 753
- Timeouts: 647
- SQLite errors: 66
- Unsafe (REPLACE etc): 40
- Duration: ~63 min, mean latency 573 ms
- Status: `FULL_BIRD_GOLD_EXECUTION_AUDIT_FAILED`
- Failure details: `artifacts/formal_readiness/gold_execution_failures.jsonl`

## 7. Engineering Validation

- 120 examples, frozen as `engineering_validation_tuning`.

## 8. Formal Validation

- 256 examples, DB-stratified, seed 20260818, frozen.
- File: `data/bird/splits/formal_validation.jsonl`

## 9. Formal Train

- 6225 examples, frozen.

## 10. Leakage Audit

- All ID overlaps = 0
- Duplicate normalized question warnings = 1
- See `artifacts/formal_readiness/leakage_report.json`

## 11. Final SELECT-500 Manifest

- 500 rows
- Exposed Mini-Dev: 20
- Unexposed: 480

## 12. Mini-Dev Exposure

Frozen.

## 13. Formal Protocol Lock

See `artifacts/formal_readiness/formal_protocol_lock.json`.

## 14. GRPO Instrumentation P1 Fixes

- Strong confirmation gate now inherits strict gate.
- RewardSampleLogger now records optimizer_step/completion_index/stop_reason/completion_tokens/sql_block_count schema fields.

## 15. GPTQ Environment Status

- Historical quantization: PASS
- Load validation: ENV BLOCKED (CUDA_HOME/nvcc missing)

## 16. Repository / GitHub Cleanup

TBD.

## 17. Fresh Clone Final Commit

TBD.

## 18. CPU Regression

TBD.

## 19. Remaining Limitations

- Gold execution audit has 753 failures (timeouts/sqlite/unsafe) -> formal data gate FAILED.
- Airline DB hash conflict pending manual review.
- GPTQ load env blocker.

## 20. Final Status

PROJECT STATUS: `FULL_BIRD_GOLD_EXECUTION_AUDIT_FAILED`

All other data prep steps completed:
- Full DB registry complete (80/80 resolved, 1 hash conflict on airline)
- Formal splits frozen (120/256/6225)
- Leakage clean
- Final SELECT-500 manifest frozen
- GPTQ remains environment-blocked (warning only)
