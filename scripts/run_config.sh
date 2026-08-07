#!/usr/bin/env bash
# Run a reproduction config in a way that is independent of the caller's CWD.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_config.sh CONFIG [run_experiment arguments...]

Environment overrides:
  PYTHON_BIN    Python executable (default: .venv/bin/python, then python3)
  RUN_ID        Identifier stored with this run (default: UTC timestamp)
  OUTPUT_ROOT   Results root (default: outputs/results)
  LIMIT         Optional number of examples to process
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

CONFIG="$1"
shift
if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${REPO_ROOT}/${CONFIG}"
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config file not found: ${CONFIG}" >&2
  exit 2
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="${PYTHON_BIN}"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON="python3"
fi
if ! command -v "${PYTHON}" >/dev/null 2>&1 && [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: Python executable not found: ${PYTHON}" >&2
  exit 2
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/results}"
if [[ "${OUTPUT_ROOT}" != /* ]]; then
  OUTPUT_ROOT="${REPO_ROOT}/${OUTPUT_ROOT}"
fi
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${OUTPUT_ROOT}/_launcher_logs"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
mkdir -p "${LOG_DIR}"

command=("${PYTHON}" -m repro.run_experiment --config "${CONFIG}" --output-root "${OUTPUT_ROOT}" --run-id "${RUN_ID}")
if [[ -n "${LIMIT:-}" ]]; then
  command+=(--limit "${LIMIT}")
fi
command+=("$@")

printf 'Run ID: %s\nConfig: %s\nOutput root: %s\nCommand:' "${RUN_ID}" "${CONFIG}" "${OUTPUT_ROOT}" | tee -a "${LOG_FILE}"
printf ' %q' "${command[@]}" | tee -a "${LOG_FILE}"
printf '\n' | tee -a "${LOG_FILE}"

cd "${REPO_ROOT}"
set +e
"${command[@]}" 2>&1 | tee -a "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e
if [[ ${status} -ne 0 ]]; then
  printf 'Run failed with exit status %s. See %s\n' "${status}" "${LOG_FILE}" | tee -a "${LOG_FILE}" >&2
  exit "${status}"
fi

printf 'Run completed successfully. Launcher log: %s\n' "${LOG_FILE}" | tee -a "${LOG_FILE}"
