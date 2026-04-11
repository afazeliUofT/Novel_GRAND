#!/bin/bash
set -euo pipefail

REPO_DIR="/home/rsadve1/scratch/Novel_GRAND"
PYBIN="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin/python3"

cd "${REPO_DIR}"

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

if [[ -e .venv-fir ]]; then
  echo "ERROR: ${REPO_DIR}/.venv-fir already exists."
  echo "Remove or rename it before rerunning this bootstrap."
  exit 1
fi

echo "Using Python interpreter: ${PYBIN}"
"${PYBIN}" -V

"${PYBIN}" -m venv .venv-fir
source .venv-fir/bin/activate
hash -r

python -m pip install --upgrade pip setuptools wheel

# CPU-only PyTorch from the official CPU wheel index
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0

# Project requirements
python -m pip install --only-binary=:all: -r env/requirements-fir.txt

# Consistency check
python -m pip check

# Login-node verification
python tools/verify_sionna_env.py | tee probe_outputs/venv_verify_login.txt

# Exact frozen environment for reproducibility
python -m pip freeze | sort > probe_outputs/venv_fir_freeze.txt

echo
echo "Bootstrap complete."
echo "Login verification:  probe_outputs/venv_verify_login.txt"
echo "Frozen packages:     probe_outputs/venv_fir_freeze.txt"
