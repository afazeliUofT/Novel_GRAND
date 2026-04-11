#!/bin/bash
set -euo pipefail

REPO_DIR="/home/rsadve1/scratch/Novel_GRAND"

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

source "${REPO_DIR}/.venv-fir/bin/activate"
hash -r
