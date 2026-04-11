#!/bin/bash
# Safe interactive smoke wrapper. This script does not alter the caller's shell
# options and always leaves a log behind, even on failure.

set -u

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd -P)"
LOG_DIR="${REPO_DIR}/probe_outputs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_DIR}/check_tags_package_${TS}.log"

{
  echo "[info] repo_dir=${REPO_DIR}"
  echo "[info] timestamp=$(date -Is)"
  echo "[info] running smoke test"
} | tee "${LOG_PATH}"

(
  cd "${REPO_DIR}" || exit 1
  # shellcheck disable=SC1091
  source env/activate_fir.sh || exit 1
  python -m novel_grand.scripts.smoke --config configs/fir_smoke.yaml
) >>"${LOG_PATH}" 2>&1
RC=$?

if [[ ${RC} -eq 0 ]]; then
  echo "[ok] smoke test passed. Log: ${LOG_PATH}"
else
  echo "[error] smoke test failed with exit code ${RC}. Log: ${LOG_PATH}" >&2
fi

exit ${RC}
