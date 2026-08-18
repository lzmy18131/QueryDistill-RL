# Reward design

## Components and ranges

| Component | Range | Logic |
| --- | --- | --- |
| format | `[-0.4, +0.05]` | protocol compliance: single `<sql>` block, `<plan>` when required |
| parse | `[-0.4, +0.05]` | extracted SQL is one valid statement |
| safety | `{+0.05, -1.0}` | AST safety decision; unsafe SQL is never executed |
| execution | `{0.0, +0.1}` | candidate ran successfully inside the sandbox |
| correctness | `{0.0, +0.25, +1.0}` | result equivalence: full / empty-structural / wrong |

Total = clamp(sum, min=-1.0). Unsafe SQL is pinned to exactly `-1.0` no matter
how pretty the surrounding text is.

## Why correctness dominates

* Correct SQL: `0.05 + 0.05 + 0.05 + 0.1 + 1.0 = 1.25`.
* Wrong but beautiful SQL: at most `0.25`.
* Malformed output: negative.
* Unsafe output: `-1.0` (hard).

A policy therefore cannot optimize formatting/safety positives while ignoring
execution equivalence.

## Anti-hacking defenses

* `SELECT 1`, `SELECT NULL`, constant queries, schema-independent queries and
  always-empty queries fail multiset comparison against non-empty gold.
* Both-empty results are capped (see `docs/RESULT_EQUIVALENCE.md`).
* Duplicate `<sql>` tags: parser refuses to execute anything.
* Hidden multiple statements: AST layer rejects; safety reward -1.0.
* Comment tricks: comments cannot create executable statements; the verifier
  still requires the right result.
* Expensive Cartesian products / recursive CTEs: execution has
  `max_execution_ms`, a progress handler, a watchdog `interrupt()` and process
  termination; timeouts earn no execution reward and can never earn
  correctness.
* Every component is logged per rollout (`reward_samples.jsonl`) with a full
  trace (parse/safety/execution/verification).

## What is deliberately not done

* No random reward, no hard-coded reward, no fake reward in the GPU smoke path.
* No learned reward model (unnecessary for verifiable SQL).
