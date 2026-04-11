#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

source "${REPO_DIR}/.venv-fir/bin/activate"
hash -r
