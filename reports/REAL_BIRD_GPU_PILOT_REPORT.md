# REAL BIRD GPU PILOT REPORT

Project: **QueryDistill-RL**
Date: 2026-08-17
Status:

```text
PROJECT STATUS: GRPO_PILOT_NEEDS_TUNING
```

This report covers the first real-GPU BIRD pilot with **Qwen3-4B Teacher**,
**Qwen3-0.6B-Base student**, real SQLite rewards, real SFT/GRPO/GPTQ runs, and
the locked 20-example BIRD Mini-Dev evaluation pilot.

> This is a pilot, not a formal benchmark. No statistical significance is
> claimed from 20 Mini-Dev examples.

---

## A. Teacher

- Teacher: `Qwen/Qwen3-4B`, revision `main`
- Inference: 4-bit NF4, bfloat16, `enable_thinking=False`, Qwen chat template,
  SQL-only protocol (`<sql>...</sql>`)
- Data: real BIRD filtered train, deterministic 20-example subset
  - `artifacts/pilot/teacher/pilot_ids.json`
  - dataset sha256: `b9e85cdc0850aac23ac3ec7bd4e2d63ea1892e7d42f729b157a9bf2328c2d8db`
- Candidate count: **2 per example**

| Metric | Value |
| --- | --- |
| Requested examples | 20 |
| Examples generated | 20 |
| Generated candidates | 40 |
| Mean candidates / example | 2.0 |
| Parse-valid rate | 100.0% |
| Safe-SQL rate | 95.0% |
| Execution-success rate | 67.5% |
| Strict-verified candidate rate | 25.0% (10/40) |
| Verified example count | 5 |
| Verified example coverage | 25.0% (5/20) |
| Generation latency | 695.14 s |
| Peak VRAM | 4873 MiB |

Teacher error summary (per candidate):

| Bucket | Count |
| --- | --- |
| Verified | 10 |
| Wrong result | 17 |
| Execution failed | 11 |
| Unsafe | 2 |

Artifacts:
- `artifacts/pilot/teacher/manifest.json`
- `artifacts/pilot/teacher/candidates.jsonl`
- `artifacts/pilot/teacher/verified.jsonl`
- `artifacts/pilot/teacher/metrics.json`
- `artifacts/pilot/teacher/run.log`

Because `verified_example_count > 0`, the pilot continued.

---

## B. Paired dataset

Built from real verified teacher records only; **no gold fallback**.

- `artifacts/pilot/paired/paired_manifest.json`
- Requested: **20**
- Paired: **5**
- Coverage: **0.25**
- Dropped IDs: 15 examples with no verified teacher candidate (listed in
  `paired_manifest.json`).

Paired example IDs (exact subset used by both SFT arms):

```text
bird-train-book_publishing_company-38016265
bird-train-book_publishing_company-14173382
bird-train-book_publishing_company-28802977
bird-train-book_publishing_company-30067580
bird-train-book_publishing_company-93064276
```

Gold and Distilled identical-subset proof:
- `artifacts/pilot/gold_sft/dataset/train.json` has **5** rows.
- `artifacts/pilot/distilled_sft/dataset/train.json` has **5** rows.
- Both SFT artifact manifests reference the same
  `artifacts/pilot/paired/paired_manifest.json`.
- Both arms use `data/bird/examples/train_pilot20.jsonl` as source and filter by
  the same `example_ids`.

---

## C. SFT pilots

Both arms:
- Qwen3-0.6B-Base + QLoRA NF4
- Paired subset (5 examples)
- 2 optimizer steps, batch size 1
- max_seq_length 1024

### Gold-SFT

| Metric | Value |
| --- | --- |
| Loss (step 1, 2) | 0.8800, 1.1858 |
| Grad norm (step 1, 2) | 3.649, 3.189 |
| Training duration | 14.20 s |
| Peak VRAM | 5928 MiB |
| Adapter sha256 | `f3d4a2fd9f8224e806625ea1e1089930bce36b212fcee5e422bf436ecda1a250` |
| Checkpoint | `checkpoints/pilot/gold_sft` |

### Distilled-SFT

| Metric | Value |
| --- | --- |
| Loss (step 1, 2) | 0.8478, 0.9310 |
| Grad norm (step 1, 2) | 4.553, 3.255 |
| Training duration | 10.35 s |
| Peak VRAM | 5888 MiB |
| Adapter sha256 | `43daef3f0c4e2d83182da235891bb67c124f4ba7a578ae248913f48b01d75d59` |
| Checkpoint | `checkpoints/pilot/distilled_sft` |

Artifacts:
- `artifacts/pilot/gold_sft/metrics.json`, `resolved_config.yaml`
- `artifacts/pilot/distilled_sft/metrics.json`, `resolved_config.yaml`

---

## D. Pilot Evaluation (locked 20 BIRD Mini-Dev)

All internal metrics use strict execution equivalence. Official BIRD EX uses the
downstream official evaluator; missing/format-invalid predictions are counted as
wrong by filling a non-equivalent placeholder (`SELECT 1`) so all 20 questions
have a prediction.

| Model | format_valid | parse_valid | exec_success | internal EX | official BIRD EX |
| --- | --- | --- | --- | --- | --- |
| Base | 40.0% | 35.0% | 15.0% | 0.0% | 0.0% |
| Gold-SFT | 15.0% | 10.0% | 10.0% | 0.0% | 0.0% |
| Distilled-SFT | 20.0% | 20.0% | 15.0% | 0.0% | 0.0% |
| GRPO (from Distilled-SFT) | 15.0% | 10.0% | 10.0% | 0.0% | 0.0% |
| GRPO GPTQ INT4 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

Observations only (no significance claims):
- All pilots had low format validity, causing most predictions to be non-executable.
- The 2-step SFT/GRPO pilots were not enough to produce reliable BIRD SQL output.
- No “Distillation improves...” statement is made.

Artifacts:
- `artifacts/pilot/eval/*.json`
- `artifacts/pilot/eval/*_official/*`

---

## E. GRPO pilot

- Initialization source: **Distilled-SFT adapter**
  - `checkpoints/pilot/distilled_sft`
  - adapter sha256 `43daef3f0c4e2d83182da235891bb67c124f4ba7a578ae248913f48b01d75d59`
- Config: `configs/experiment/bird_pilot_grpo.yaml`
  - `num_generations=2`, batch size 1, max_steps 2
  - `max_prompt_length=1024`, `max_completion_length=128`
- Output: `checkpoints/pilot/grpo`
- Peak VRAM: **4515 MiB**

Reward evidence (`checkpoints/pilot/grpo/reward_samples.jsonl`):
- Real SQLite environment: yes.
- Both logged rollouts are for `bird-train-book_publishing_company-27870599`.
- Reward = `-0.8` each (`format=-0.4`, `parse=-0.4`, safety/execution/correctness=0).
- `parse_ok=false`, no SQL execution because the outputs were not parseable.

GRPO metrics:

| Metric | Value |
| --- | --- |
| Reward mean | -0.8 |
| Reward std values | [0.0] |
| Zero-std groups / total groups | 1 / 1 |
| Grad norm values | [0.0, 0.0] |
| Parameter fingerprint before | `282ec0d18117fbb39d2483e28a3860ab63ca5c86e62596a5bea25c503a4f261f` |
| Parameter fingerprint after | `bcd3af331d442281ffb893e796c35f2c64b70c6fe4b2bf99be1569ed89aa5112` |
| Fingerprint changed | true |
| Status | `GRPO_INTEGRATION_PASS` |
| Learning signal | `GRPO_LEARNING_SIGNAL_INSUFFICIENT` |

Cause analysis:
- The student generated malformed completions for the sampled rollout, so all
  rewards were identical (`reward_std=0`).
- With zero reward variance, TRL produced zero gradients (`grad_norm=0`).
- The parameter fingerprint changed (optimizer/state touched tensors) but this
  is not a useful learning signal.
- No reward variance was fabricated.

Per the protocol, full GRPO is **not** started.

---

## F. GPTQ pilot

Executed because a real GRPO artifact was produced, even though the GRPO
learning signal was insufficient.

- Input GRPO artifact: `checkpoints/pilot/grpo/adapter`
  - adapter sha256 `33d3102bad88dd280142f9a53dd99236aa906cc535857e0af7db0711515713ae`
- Merged model: `checkpoints/pilot/merged`
- GPTQ INT4 output: `checkpoints/pilot/gptq/quantized`
- Calibration: train-only (`data/bird/examples/train_pilot20.jsonl`, 16 samples)

| Metric | GRPO pre-GPTQ | GRPO GPTQ INT4 |
| --- | --- | --- |
| Internal EX | 0.0% | 0.0% |
| Official BIRD EX | 0.0% | 0.0% |
| Model size | 1.11 GB (merged bf16) | 516.27 MB (quantized) |
| Peak VRAM (eval) | not separately captured | 2227 MiB |
| Load time | 244.94 s | 226.21 s |
| Mean generation latency | 14683 ms | 9123 ms |

No claim is made that INT4 is faster or slower; the latency difference is a
single 20-example pilot observation and is confounded by CPU execution and
different load/compilation paths.

---

## G. Honest limitations

1. **Tiny sample sizes**: 5 verified training examples, 20 Mini-Dev questions,
   40 teacher candidates. No statistical conclusions.
2. **GRPO learning signal insufficient**: reward_std=0, grad_norm=0. The pilot
   cannot be used to claim RL improvement.
3. **Official EX 0.0** is largely driven by format/parse failures, not by
   execution-equivalence on well-formed SQL.
4. **Evaluation ran on CPU** in the current harness path (models were loaded
   without `device_map="cuda"` by the evaluation CLI), which makes latencies
   environment-specific, not comparable to production GPU serving.
5. **GPTQ comparison is observational**, not a latency benchmark.
6. **vLLM**: `VLLM_NOT_SMOKE_VERIFIED` — no live vLLM smoke was run this round.
7. **Teacher strict verification is low** (25% example coverage); the small
   paired dataset reflects real teacher behavior, not cherry-picking.

---

## Artifacts

```text
artifacts/pilot/teacher/
artifacts/pilot/paired/
artifacts/pilot/gold_sft/
artifacts/pilot/distilled_sft/
artifacts/pilot/grpo/
artifacts/pilot/gptq/
artifacts/pilot/eval/
artifacts/pilot/vllm/
reports/REAL_BIRD_GPU_PILOT_REPORT.md
```

## Final gate

- Real Teacher executed: PASS
- Teacher verified candidates > 0: PASS
- Paired Gold/Distilled dataset correct: PASS
- Gold-SFT real run PASS: PASS
- Distilled-SFT real run PASS: PASS
- Official evaluator PASS: PASS (runs; all pilot EX=0)
- GRPO initialized from Distilled-SFT: PASS
- GRPO real SQLite reward PASS: PASS
- GRPO learning signal PASS: **FAIL** (reward_std=0, grad_norm=0)
- GPTQ from GRPO PASS: PASS (artifact produced)

```text
PROJECT STATUS: GRPO_PILOT_NEEDS_TUNING
```
