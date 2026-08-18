#!/usr/bin/env bash
# CPU-only quality gates (safe for CI and native Windows Git Bash).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -x ".venv/Scripts/python.exe" ]]; then
  PY=".venv/Scripts/python.exe"
  RUFF=".venv/Scripts/ruff.exe"
elif [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
  RUFF=".venv/bin/ruff"
else
  PY="python"
  RUFF="ruff"
fi

"${PY}" -m compileall -q src tests
"${RUFF}" check .
"${RUFF}" format --check .
"${PY}" -m pytest -q -m "not gpu"
