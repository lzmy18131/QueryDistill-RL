#!/usr/bin/env bash
# vLLM serving launcher for the project (run inside WSL2 Ubuntu).
# Usage: bash scripts/serve_vllm.sh [config yaml] [extra vllm args...]
#
# The serve environment is intentionally separate from the training env:
#   .venv-serve/  (vLLM + its own torch/CUDA pins)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="${1:-configs/serving/vllm_smoke.yaml}"
shift || true

# Load cache env overrides if present (never exported to the system).
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -x ".venv-serve/bin/python" ]]; then
  PY=".venv-serve/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
  echo "WARNING: using the training venv for vLLM (serve env .venv-serve not found)" >&2
  PY=".venv/bin/python"
else
  PY="python"
fi

echo "==> Project: ${PROJECT_ROOT}"
echo "==> Python:  ${PY}"
echo "==> Config:  ${CONFIG}"

"${PY}" -m querydistill.cli serve-vllm --config "${CONFIG}" --print-command
COMMAND=$("${PY}" - <<'PY'
import sys, yaml
sys.path.insert(0, "src")
from querydistill.serving.vllm import VLLMServeConfig, build_server_command
from querydistill.config import Settings
settings = Settings.load()
path = settings.project_root / sys.argv[1]
config = VLLMServeConfig(**{k: v for k, v in yaml.safe_load(path.read_text()).items() if k in VLLMServeConfig.__dataclass_fields__})
config.model_path = str((settings.project_root / config.model_path).resolve())
print(" ".join(build_server_command(config)))
PY
"${CONFIG}")

echo "==> Executing: ${COMMAND} $*"
# shellcheck disable=SC2086
exec ${COMMAND} "$@"
