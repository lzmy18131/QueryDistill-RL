# Changelog

## 0.1.0 - FIRST IMPLEMENTATION ROUND (2026-08-17)

* Project skeleton: config system with D-drive enforcement, `.env.example`,
  hardware doctor, CLI.
* SQL safety: sqlglot AST allowlist + read-only SQLite authorizer + progress
  handler + watchdog interrupt + spawn-process isolation + max_rows.
* `SQLExecutionEnvironment` with db_id allowlist registry.
* `ResultEquivalenceVerifier` (multiset / ordered / empty-structural rules).
* Output protocol parser (`<plan>` / `<sql>`, fences, duplicate-tag defense).
* Reward stack: format/parse/safety/execution/correctness + composite.
* Reward-hacking test suite.
* Tiny synthetic fixtures: 3 SQLite DBs + 42 examples across 9 query types.
* Dataset schema (Example / DistillationRecord), LeakageGuard, data audit.
* Resumable teacher distillation pipeline (mock + transformers backend).
* LLaMA-Factory QLoRA backend (YAML generation, dataset registration, target
  module detection, log collection, resume wrapper).
* TRL GRPO backend with the real SQLite reward environment and per-rollout
  reward logging.
* GPTQModel INT4 adapter with calibration contamination guard.
* vLLM serve config + OpenAI-compatible benchmark client (TTFT/P50/P95/tokens/
  requests/peak VRAM).
* Unified evaluation harness + error buckets + McNemar / paired bootstrap.
* CPU test suite: 203 passed / 1 skipped (GPU probe) as recorded in
  `docs/FIRST_BUILD_REPORT.md`.
* Quality gates: compileall, ruff check, ruff format --check all pass.
* GPU smoke status recorded honestly per stage; see FIRST_BUILD_REPORT.
