# References

Papers and primary documentation consulted for algorithm understanding. No
paper repository training script was copied.

## Algorithms

1. DeepSeek-AI, *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in
   Open Language Models* (2024) - GRPO formulation.
   arXiv:2402.03300. Used to understand group-relative policy optimization.
2. Shao et al., *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
   Reinforcement Learning* (2025) - verifiable rewards and RL scaling.
   arXiv:2501.12948.
3. Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023) -
   NF4 / double quantization / LoRA on quantized base.
   arXiv:2305.14314.
4. Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative
   Pre-trained Transformers* (2022) - GPTQ algorithm.
   arXiv:2210.17323.
5. Hinton et al., *Distilling the Knowledge in a Neural Network* (2015) -
   knowledge distillation background. arXiv:1503.02531.
6. Schulman et al., *Proximal Policy Optimization Algorithms* (2017) - PPO,
   referenced only as the rejected baseline in docs/ALGORITHM_DECISIONS.md.
   arXiv:1707.06347.
7. Rafailov et al., *Direct Preference Optimization* (2023) - DPO, referenced
   only for the optional offline ablation. arXiv:2305.18286.
8. Li et al., *Can LLM Already Serve as A Database Interface? A BIg Bench for
   Large-Scale Database Grounded Text-to-SQLs* (BIRD benchmark, 2023) - target
   benchmark design reference, not used in the first round.
   arXiv:2305.03111.

## Tool documentation (official)

* TRL: https://huggingface.co/docs/trl (GRPOTrainer / GRPOConfig)
* PEFT: https://huggingface.co/docs/peft
* Transformers: https://huggingface.co/docs/transformers
* LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Apache-2.0)
* GPTQModel: https://github.com/ModelCloud/GPTQModel (Apache-2.0)
* vLLM: https://docs.vllm.ai
* sqlglot: https://sqlglot.com
* SQLite authorizer:
  https://www.sqlite.org/c3ref/set_authorizer.html
* SQLite security overview: https://www.sqlite.org/security.html

## Machine-specific note

All URLs were consulted for API usage only. No benchmark numbers from any
paper or repository are reproduced in this project.
