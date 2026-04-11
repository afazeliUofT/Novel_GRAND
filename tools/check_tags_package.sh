#!/bin/bash
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${REPO_DIR}"
mkdir -p probe_outputs

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="probe_outputs/check_tags_package_${STAMP}.log"
CFG="${NOVEL_GRAND_CONFIG:-configs/fir_smoke.yaml}"

echo "Running TAGS-GRAND smoke check with ${CFG}"
echo "Log: ${LOG_PATH}"

if ! source env/activate_fir.sh; then
  echo "ERROR: failed to activate env/activate_fir.sh" | tee "${LOG_PATH}"
  exit 1
fi

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -m novel_grand.scripts.smoke --config "${CFG}" >"${LOG_PATH}" 2>&1
RC=$?

if [[ ${RC} -eq 0 ]]; then
  echo "Smoke check passed. See ${LOG_PATH}"
else
  echo "Smoke check failed with exit code ${RC}. See ${LOG_PATH}"
fi

exit ${RC}
