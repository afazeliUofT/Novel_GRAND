#!/bin/bash
set -uo pipefail

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT=20

section () {
  echo
  echo "================================================================"
  echo "$1"
  echo "================================================================"
}

section "BASIC CONTEXT"
echo "timestamp: $(date -Is)"
echo "hostname : $(hostname)"
echo "pwd      : $(pwd)"
echo "user     : ${USER:-unknown}"
echo "shell    : ${SHELL:-unknown}"

section "OS / GLIBC / CPU / MEMORY / DISK"
uname -a || true
echo
cat /etc/os-release || true
echo
getconf GNU_LIBC_VERSION || true
ldd --version 2>&1 | head -n 1 || true
echo
lscpu || true
echo
nproc --all || true
echo
free -h || true
echo
df -h . || true

section "SLURM / MODULES"
env | grep -E '^(SLURM|SBATCH)_' | sort || true
echo
if command -v sinfo >/dev/null 2>&1; then
  sinfo -o "%P %c %m %G %l %D %N" || true
else
  echo "sinfo not found"
fi
echo
if command -v module >/dev/null 2>&1; then
  echo "--- module --version ---"
  module --version 2>&1 || true
  echo
  echo "--- module list ---"
  module list 2>&1 || true
  echo
  echo "--- module -t avail python ---"
  module -t avail python 2>&1 || true
  echo
  echo "--- module -t avail gcc ---"
  module -t avail gcc 2>&1 || true
  echo
  echo "--- module -t avail cuda ---"
  module -t avail cuda 2>&1 || true
else
  echo "module command not found in this shell"
fi

section "TOOLCHAIN PATHS"
for x in python python3 pip pip3 gcc g++ make cmake git curl wget; do
  printf '%-10s -> ' "$x"
  command -v "$x" || true
done

PY="$(command -v python3 || command -v python || true)"

section "PYTHON CORE"
if [[ -z "${PY}" ]]; then
  echo "No python interpreter found in PATH"
else
  "$PY" - <<'PY'
import importlib.util
import json
import os
import platform
import ssl
import sys
import sysconfig

mods = [
    "venv", "ensurepip", "ssl", "sqlite3", "bz2", "lzma",
    "pip", "setuptools", "wheel", "packaging"
]
status = {m: (importlib.util.find_spec(m) is not None) for m in mods}

print("sys.executable :", sys.executable)
print("sys.version    :", sys.version.replace("\n", " "))
print("implementation :", platform.python_implementation())
print("platform       :", platform.platform())
print("machine        :", platform.machine())
print("SOABI          :", sysconfig.get_config_var("SOABI"))
print("EXT_SUFFIX     :", sysconfig.get_config_var("EXT_SUFFIX"))
print("OpenSSL        :", ssl.OPENSSL_VERSION)
print("PYTHONPATH     :", os.environ.get("PYTHONPATH"))
print("module_status  :", json.dumps(status, indent=2))
PY
fi

section "PIP INFO"
if [[ -n "${PY}" ]]; then
  "$PY" -m pip --version || true
  echo
  "$PY" -m pip debug --verbose || true
fi

section "TEMP VENV SMOKE TEST"
if [[ -n "${PY}" ]]; then
  rm -rf probe_outputs/.tmp_venv_probe
  "$PY" -m venv probe_outputs/.tmp_venv_probe && \
  probe_outputs/.tmp_venv_probe/bin/python -m pip --version && \
  probe_outputs/.tmp_venv_probe/bin/python - <<'PY'
import ssl, sys
print("tmp_venv_python :", sys.executable)
print("tmp_venv_ssl    :", ssl.OPENSSL_VERSION)
PY
  rc=$?
  rm -rf probe_outputs/.tmp_venv_probe
  exit_code=$rc
  if [[ $exit_code -ne 0 ]]; then
    echo "Temporary venv creation FAILED"
  else
    echo "Temporary venv creation SUCCEEDED"
  fi
fi

section "PYPI / WHEEL ACCESS"
if [[ -n "${PY}" ]]; then
  "$PY" - <<'PY'
import json
import ssl
import urllib.request

packages = ["sionna", "sionna-no-rt", "torch", "tensorflow", "tensorflow-cpu"]

for pkg in packages:
    url = f"https://pypi.org/pypi/{pkg}/json"
    try:
        with urllib.request.urlopen(url, timeout=15, context=ssl.create_default_context()) as r:
            data = json.load(r)
        print(f"{pkg:16s} latest={data['info']['version']}")
    except Exception as e:
        print(f"{pkg:16s} FAIL {repr(e)}")

extra_urls = [
    "https://download.pytorch.org/whl/cpu/",
]
for url in extra_urls:
    try:
        with urllib.request.urlopen(url, timeout=15, context=ssl.create_default_context()) as r:
            _ = r.read(512)
        print(f"URL OK   {url}")
    except Exception as e:
        print(f"URL FAIL {url} {repr(e)}")
PY
fi

section "PIP DRY-RUN RESOLVER"
if [[ -n "${PY}" ]]; then
  if "$PY" -m pip help install 2>/dev/null | grep -q -- '--dry-run'; then
    for pkg in "sionna-no-rt" "sionna" "torch" "tensorflow" "tensorflow-cpu"; do
      echo
      echo ">>> DRY RUN: ${pkg}"
      "$PY" -m pip install \
        --dry-run \
        --ignore-installed \
        --only-binary=:all: \
        "$pkg" || true
    done
  else
    echo "pip on this system does not support --dry-run"
  fi
fi

section "ENVIRONMENT VARIABLES OF INTEREST"
env | grep -E '^(OMP|MKL|OPENBLAS|NUMEXPR|VECLIB|TF_|PYTORCH_|CUDA|LD_LIBRARY_PATH|LIBRARY_PATH|CPATH|C_INCLUDE_PATH|CPLUS_INCLUDE_PATH)=' | sort || true
