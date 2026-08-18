# Algorithm decisions

## Why GRPO is the primary RL method

1. **Verifiable reward exists.** Text-to-SQL has a real execution environment:
   a candidate either produces the gold result (modulo documented equivalence
   rules) or it does not. No learned reward model is needed, which removes a
   whole training pipeline and a source of reward hacking.
2. **No critic / value model.** GRPO estimates a group-relative baseline from
   multiple rollouts of the same prompt instead of training an independent
   value network. On one RTX 4060 Laptop (8 GB VRAM), removing the critic is
   a large memory and implementation win.
3. **Group-relative signal.** Multiple SQL rollouts for the same prompt are
   ranked against each other; the policy learns to prefer executions that the
   verifier accepts over pretty-but-wrong alternatives.
4. **Online verifier contact.** Every rollout passes through the parser,
   safety layer and SQLite environment during training, so the policy is
   optimized against the exact metric used at evaluation time.

## Why PPO is not implemented

* PPO needs a value model/critic and GAE, roughly doubling memory and moving
  parts on already-constrained consumer hardware.
* With an objective, verifiable reward and small group rollouts, the
  group-relative GRPO estimator is the more economical choice.
* PPO is not implemented anywhere in this repository (README states this).

## Why standard DPO is not the primary RL

* Standard DPO is an **offline** preference optimization: it learns from a
  fixed chosen/rejected dataset.
* This task has a live execution environment; online rollouts can use the
  verifier directly instead of distilling preferences into pairs.
* Constructing DPO pairs requires choosing a rejection strategy; an online
  method naturally samples its own negatives.

## DPO optional ablation

DPO is kept as an optional offline baseline only:

* chosen = execution-correct SQL
* rejected = execution-wrong SQL

Future comparison: Offline Preference Optimization (Distilled-SFT + DPO) vs
Online Verifiable RL (Distilled-SFT + GRPO), with identical data budget, base
checkpoint and eval split. Plan: `docs/DPO_ABLATION_PLAN.md`.

## Quantization strategy

* QLoRA (4-bit NF4 training-time) makes SFT/GRPO fit in 8 GB VRAM.
* Deployment compression is a separate GPTQ INT4 step after LoRA merge.
* These two quantization stages are never conflated.
