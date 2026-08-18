# Scaling notes

## Why this machine does not need DeepSpeed for the planned experiments

* Student is Qwen3-0.6B-Base (~1.2 GB in BF16).
* QLoRA (4-bit NF4 base + LoRA adapters) keeps optimizer states tiny; Adam
  states exist only for adapter parameters.
* Batch sizes are 1 with gradient accumulation; activation memory is bounded
  by `max_seq_length <= 1024` and gradient checkpointing in SFT.
* GRPO uses num_generations=2-4 and short completions in smoke mode.

ZeRO sharding adds cross-process overhead with no benefit at this scale, and
the machine has one GPU. Therefore: no DeepSpeed in round 1, no DeepSpeed
claim anywhere.

## Future scale-out

`configs/deepspeed/zero2_example.yaml` is a **future example only**. It was
never executed. If the project moves to a larger student or multi-GPU box, the
correct move is to re-benchmark ZeRO-2 vs plain QLoRA rather than assume.

## Memory budgeting rules

* Teacher and student never co-reside in VRAM (teacher unload path is enforced
  in `src/querydistill/distillation/backends.py`).
* vLLM lives in a separate environment so its torch/CUDA wheel pins cannot
  damage the training environment.
* All caches live on D: (`D:\LLMCache`); C-drive cache paths are warnings.
