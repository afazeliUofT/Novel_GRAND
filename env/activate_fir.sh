#!/bin/bash
# Source this file to activate the FIR project virtualenv.
# Intentionally do not enable set -e/-u/-o pipefail here because this file is
# sourced into the caller's shell and must not change interactive shell error
# handling.

_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
_ACTIVATE_PATH="${_REPO_DIR}/.venv-fir/bin/activate"

export PYTHONNOUSERSITE=1
unset PYTHONPATH 2>/dev/null || true

if [[ ! -f "${_ACTIVATE_PATH}" ]]; then
  echo "ERROR: virtualenv activation script not found at ${_ACTIVATE_PATH}" >&2
  echo "Create the FIR venv first before sourcing env/activate_fir.sh." >&2
  unset _REPO_DIR _ACTIVATE_PATH
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "${_ACTIVATE_PATH}"
hash -r 2>/dev/null || true
unset _REPO_DIR _ACTIVATE_PATH
