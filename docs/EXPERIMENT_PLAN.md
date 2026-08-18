# Experiment plan (full experiment, after code review)

This document is the roadmap for the rounds that may only start **after** the
first code review passes. Nothing here claims to have run.

## Arms

1. Student Base (Qwen3-0.6B-Base)
2. Gold-SFT (target = gold SQL)
3. Distilled-SFT (target = verified teacher SQL)
4. Distilled-SFT + GRPO (real SQLite reward)
5. GRPO BF16/FP16 deployment
6. GRPO GPTQ INT4 deployment

## Fixed protocol for comparability

* Same evaluation harness (`src/querydistill/evaluation`) for every arm.
* Same held-out test split; dev only for early stopping.
* Same training budget for Gold-SFT and Distilled-SFT.
* GRPO starts from the Distilled-SFT adapter.
* GPTQ calibration uses train/calibration splits only (enforced in code).

## Metrics

Primary: Execution Accuracy. Secondary: Valid SQL Rate, Execution Success Rate,
Unsafe SQL Rate, Exact Match (secondary), latency. Deployment adds VRAM, model
size, throughput, TTFT (concurrency 1/4/8).

## Statistics

* Gold-SFT vs Distilled-SFT: McNemar exact + paired bootstrap CI.
* Distilled-SFT vs GRPO: McNemar exact + paired bootstrap CI.
* Tiny synthetic fixture results are **never** used for statistical claims.

## Teacher protocol

Teacher generates offline candidates -> verified against gold execution ->
persisted dataset -> teacher unloaded -> gc -> `torch.cuda.empty_cache()` ->
student training. Teacher and student never share the GPU. The 4B teacher is
not downloaded until the student pipeline fully passes and only 1-3 smoke
samples are needed.
