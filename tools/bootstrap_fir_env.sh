#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYBIN="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin/python3"

cd "${REPO_DIR}"
mkdir -p probe_outputs

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

if [[ -e .venv-fir ]]; then
  echo "ERROR: ${REPO_DIR}/.venv-fir already exists."
  echo "Remove or rename it before rerunning this bootstrap."
  exit 1
fi

echo "Using repository: ${REPO_DIR}"
echo "Using Python interpreter: ${PYBIN}"
"${PYBIN}" -V

"${PYBIN}" -m venv .venv-fir
source .venv-fir/bin/activate
hash -r

python -m pip install --upgrade pip setuptools wheel

# Install NumPy first so torch and later probes import cleanly
python -m pip install --only-binary=:all: numpy==2.2.6

# Install CPU-only PyTorch version that satisfies Sionna 2.0.1
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  torch==2.9.1

# Install wheel-based runtime deps
python -m pip install --only-binary=:all: -r env/requirements-fir.txt

# Install Sionna itself without re-resolving torch/deps
python -m pip install --no-deps sionna-no-rt==2.0.1

# Sanity checks and reproducibility artifacts
python -m pip check

python -m pip show \
  torch sionna-no-rt numpy scipy h5py matplotlib importlib-resources \
  pandas PyYAML tqdm psutil | tee probe_outputs/pip_show_fir.txt

python tools/verify_sionna_env.py | tee probe_outputs/venv_verify_login.txt
python -m pip freeze | sort > probe_outputs/venv_fir_freeze.txt

echo
echo "Bootstrap complete."
echo "Login verification: probe_outputs/venv_verify_login.txt"
echo "Pip package report: probe_outputs/pip_show_fir.txt"
echo "Frozen packages: probe_outputs/venv_fir_freeze.txt"
