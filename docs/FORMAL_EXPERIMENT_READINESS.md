# Formal Experiment Readiness

Status: PREPARATION ONLY — no formal training/evaluation started.

## Full train source

- Dataset: `birdsql/bird23-train-filtered`
- Rows: 6601
- Source provenance: `artifacts/formal_readiness/source_provenance.json`
- Raw SHA256: `2ee64aa593e1adc59be7d5b9ce2395372a8ee27fdd94a3fb4d2fe09f3cd34bc8`

## DB coverage

- Required train DBs: 69
- Required eval DBs: 11
- Union DBs: 80
- Missing DBs: 0
- One hash conflict on `airline` (existing vs official) -> manual review required.
- DB registry: `artifacts/formal_readiness/db_registry_manifest.json`

## Splits

- Engineering validation: 120 (frozen)
- Formal validation: 256 (DB-stratified, deterministic seed 20260818, frozen)
- Formal train: 6225
- Sum: 6601; overlap: 0

## Leakage

- Formal train / engineering / formal validation / Mini-Dev overlaps: 0
- Duplicate normalized question warnings: 1

## Final eval

- Final SELECT-only Mini-Dev manifest: 500
- Historically exposed engineering Mini-Dev: 20
- Unexposed: 480
- No model evaluation has been run in this phase.

## Historical pilot proof

- Teacher / paired SFT / protocol-correct GRPO / GPTQ quantization are pilot PASS.
- See `reports/PRE_FULL_HARDENING_REPORT.md`.

## GPTQ blocker

- GPTQ load validation remains blocked by missing CUDA_HOME/nvcc in WSL.
- This is an environment warning, not a Full-BIRD data gate blocker.
