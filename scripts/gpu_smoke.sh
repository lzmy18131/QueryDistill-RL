#!/usr/bin/env bash
# GPU smoke orchestration (Round 2.1).
#
# Run inside WSL2 Ubuntu with a CUDA environment. Uses separate venvs:
#   .venv-sft   -> LLaMA-Factory SFT smoke
#   .venv-wsl   -> inference / GRPO / GPTQ smoke
#
# The chain is intentionally train-only and fail-closed:
#   Student inference
#   Gold-SFT smoke
#   Distilled-SFT smoke (verified mock/fixture teacher targets)
#   GRPO initialized from Distilled-SFT adapter
#   GPTQ INT4 consuming the GRPO adapter
#   vLLM optional (if the serve environment is ready; otherwise NOT_VERIFIED)
#
# Mandatory failures are summed and cause a non-zero exit. vLLM is optional and
# does not count as a mandatory failure.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export HF_HOME="${HF_HOME:-/mnt/d/LLMCache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/d/LLMCache/huggingface/datasets}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1

SFT_PY=".venv-sft/bin/python"
WSL_PY=".venv-wsl/bin/python"
if [[ ! -x "${SFT_PY}" ]]; then SFT_PY=".venv/bin/python"; fi
if [[ ! -x "${WSL_PY}" ]]; then WSL_PY="${SFT_PY}"; fi

mkdir -p artifacts/smoke
MANDATORY_FAILURES=0

echo "============================================================"
echo "QueryDistill-RL Round 2.1 GPU smoke - $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "sft python: ${SFT_PY}"
echo "wsl python: ${WSL_PY}"
echo "============================================================"

run_mandatory_stage() {
  local name="$1"
  shift
  echo ""
  echo "----- [${name}] -----"
  if "$@"; then
    echo "----- [${name}] command exited 0 (see artifacts/smoke/${name}/status.json) -----"
  else
    echo "----- [${name}] command exited non-zero (see artifacts/smoke/${name}/status.json) -----"
    MANDATORY_FAILURES=$((MANDATORY_FAILURES + 1))
  fi
}

run_mandatory_stage inference \
  "${WSL_PY}" scripts/run_inference_smoke.py --model-path models/qwen3-0.6b-base

run_mandatory_stage sft \
  "${SFT_PY}" -m querydistill.cli train-sft --config configs/sft/gold_smoke.yaml

run_mandatory_stage sft-distilled \
  "${SFT_PY}" -m querydistill.cli train-sft --config configs/sft/distilled_smoke.yaml

run_mandatory_stage grpo \
  "${WSL_PY}" -m querydistill.cli train-grpo --config configs/grpo/smoke.yaml

run_mandatory_stage gptq \
  "${WSL_PY}" -m querydistill.cli quantize-gptq --config configs/quant/gptq_int4_smoke.yaml

# vLLM is optional: record honest NOT_VERIFIED instead of blocking the review.
"${WSL_PY}" -m querydistill.cli serve-vllm --config configs/serving/vllm_smoke.yaml || true

"${WSL_PY}" -m querydistill.cli report

cat > artifacts/smoke/overall_status.json <<JSON
{
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "mandatory_failures": ${MANDATORY_FAILURES},
  "status": "$([ ${MANDATORY_FAILURES} -gt 0 ] && echo FAILED || echo PASSED)"
}
JSON

echo ""
echo "GPU smoke finished. mandatory_failures=${MANDATORY_FAILURES}"
if [[ ${MANDATORY_FAILURES} -gt 0 ]]; then
  exit 1
fi
