#!/usr/bin/env bash
# GPU smoke test: checks access first, then processes a tiny FaithEval subset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi was not found. Run this smoke test on an NVIDIA GPU host." >&2
  exit 2
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "ERROR: set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) before loading gated Meta-Llama-3.1-8B-Instruct." >&2
  exit 2
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN="${HUGGINGFACE_HUB_TOKEN}"
fi

export LIMIT="${LIMIT:-5}"
export RUN_ID="${RUN_ID:-smoke_faitheval_$(date -u +%Y%m%dT%H%M%SZ)}"
CONFIG="${SMOKE_CONFIG:-configs/reproduction/faithfulrag_faitheval.json}"

echo "Running ${LIMIT} samples using ${CONFIG}. Extra arguments are forwarded to repro.run_experiment."
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PREFLIGHT_PYTHON="${PYTHON_BIN}"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PREFLIGHT_PYTHON="${REPO_ROOT}/.venv/bin/python"
else
  PREFLIGHT_PYTHON="python3"
fi
cd "${REPO_ROOT}"
"${PREFLIGHT_PYTHON}" -m repro.preflight --config "${CONFIG}" --strict
exec bash "${SCRIPT_DIR}/run_config.sh" "${CONFIG}" "$@"
