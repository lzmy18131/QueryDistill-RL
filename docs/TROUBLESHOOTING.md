# Troubleshooting

This file records environment fixes from Round 1/2 without keeping dozens of
one-off patch scripts in the repository root. Prefer the formal scripts:

- `scripts/gpu_smoke.sh` — full GPU smoke chain.
- `scripts/run_cpu_checks.sh` — compileall + ruff + pytest.
- `scripts/run_inference_smoke.py` — student inference smoke.
- `scripts/serve_vllm.sh` — vLLM serve command wrapper.
- `scripts/benchmark_vllm.py` — OpenAI-compatible benchmark client.

## WSL / CUDA

- WSL2 Ubuntu is the primary CUDA environment. Verify with:
  `wsl -d Ubuntu -- nvidia-smi`
- Native Windows is CPU-only for tests; do not run training there.
- If `torch.cuda.is_available()` is false in WSL, reinstall the exact CUDA
  wheel matching the Windows driver. Example that worked on this machine:
  `torch==2.13.0+cu126` (driver 566.24 = CUDA 12.7, so cu126 is compatible).

## Hugging Face downloads

- `huggingface.co` can be unreachable from this network while `hf-mirror.com`
  works. Set `HF_ENDPOINT=https://hf-mirror.com` and
  `HF_HUB_DISABLE_XET=1` before downloads.
- Keep all HF/torch/pip caches on D: by exporting `HF_HOME=/mnt/d/LLMCache/...`
  and the corresponding `PIP_CACHE_DIR` / `UV_CACHE_DIR`.

## LLaMA-Factory / torchaudio / torchvision

- Some LLaMA-Factory versions import `torchaudio`/`torchvision` unconditionally.
  On the text-only path these can be satisfied by minimal local stubs in the
  venv's site-packages (do not install large audio/vision stacks). The stubs are
  already applied in the working `.venv-sft`; if a fresh venv needs them, add a
  two-module stub package manually.

## vLLM

- vLLM cannot run natively on Windows. Use a dedicated WSL venv.
- The first round repeatedly hit `/tmp` ENOSPC while pip downloaded CUDA
  packages in WSL. Move pip's temp dir to D: or increase `/tmp`.
- vLLM is optional for Round 2: if the serve environment is not ready, record
  `VLLM_NOT_SMOKE_VERIFIED`; do not block the code review on it.

## SQL executor

- The safe executor uses `multiprocessing` with `spawn` on Windows. Tests that
  invoke it must run from a file (not `python - <<'PY'` from stdin) or the
  child process cannot re-import the main module.
