# DPO ablation plan (optional offline baseline, not implemented in round 1)

## Goal

Compare:

* Distilled-SFT + DPO (offline preference optimization)
* Distilled-SFT + GRPO (online verifiable RL)

## Pair construction

* chosen = execution-correct SQL
* rejected = execution-wrong SQL

Pairs come from policy/teacher rollouts that were executed and verified, so the
preference signal is grounded in the same verifier used by GRPO.

## Fairness constraints

* identical data budget (same number of examples / rollout pairs)
* identical base checkpoint (same Distilled-SFT adapter)
* identical eval split
* same evaluation harness and metrics
* same number of optimizer steps per consumed pair where practical

## What must be recorded

* pair source (teacher vs policy rollouts)
* rejection distribution (parse/safety/execution errors vs wrong results)
* chosen/rejected execution evidence (execution logs)
* final McNemar + paired bootstrap on the shared test split

## Status

Not implemented in the first round (allowed by the project spec). No DPO
results are claimed.
