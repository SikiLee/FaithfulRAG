#!/usr/bin/env bash
# Create the reproducibility virtual environment used by the wrapper scripts.
# Linux only: vLLM does not provide the supported runtime for native Windows.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  BASE_PYTHON="${PYTHON_BIN}"
elif command -v python3.10 >/dev/null 2>&1; then
  BASE_PYTHON="python3.10"
elif command -v python3 >/dev/null 2>&1; then
  BASE_PYTHON="python3"
else
  echo "ERROR: Python 3.10 or 3.11 is required but was not found." >&2
  exit 1
fi

"${BASE_PYTHON}" - <<'PY'
import sys
if sys.version_info[:2] not in {(3, 10), (3, 11)}:
    raise SystemExit(
        f"ERROR: FaithfulRAG reproduction supports Python 3.10/3.11, got {sys.version.split()[0]}"
    )
print(f"Using base Python: {sys.version.split()[0]}")
PY

if [[ ! -d "${VENV_DIR}" ]]; then
  "${BASE_PYTHON}" -m venv "${VENV_DIR}"
fi

PYTHON_BIN="${VENV_DIR}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: expected virtual-environment Python at ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m pip install --upgrade "pip==24.3.1"
"${PYTHON_BIN}" -m pip install -r "${REPO_ROOT}/requirements-repro.txt"

"${PYTHON_BIN}" - <<'PY'
import nltk
for resource in ("punkt", "punkt_tab"):
    print(f"Downloading NLTK resource: {resource}")
    nltk.download(resource, quiet=False)
PY

"${PYTHON_BIN}" - <<'PY'
import importlib
import platform
import sys

print("\nInstalled runtime")
print(f"  Python: {sys.version.split()[0]}")
print(f"  Platform: {platform.platform()}")
for package in ("torch", "transformers", "vllm", "sentence_transformers", "datasets", "nltk"):
    module = importlib.import_module(package)
    print(f"  {package}: {getattr(module, '__version__', 'unknown')}")

import torch
print(f"  torch CUDA available: {torch.cuda.is_available()}")
print(f"  torch CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
PY

cat <<EOF

Environment ready: ${VENV_DIR}
Run experiments with:
  PYTHON_BIN=${PYTHON_BIN} bash scripts/smoke_test.sh
EOF
