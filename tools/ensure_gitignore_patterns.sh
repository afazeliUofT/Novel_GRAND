#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${REPO_DIR}"

append_if_missing() {
  local pattern="$1"
  if ! grep -qxF "$pattern" .gitignore 2>/dev/null; then
    printf '%s\n' "$pattern" >> .gitignore
  fi
}

append_if_missing ''
append_if_missing '# FIR local virtualenv'
append_if_missing '.venv-fir/'
append_if_missing ''
append_if_missing '# Python build/cache artifacts'
append_if_missing '__pycache__/'
append_if_missing '*.py[cod]'
append_if_missing '*.egg-info/'
append_if_missing ''
append_if_missing '# Local overlay/package archives created during iteration'
append_if_missing 'Novel_GRAND_*overlay.zip'
append_if_missing 'Novel_GRAND_*package.zip'
append_if_missing 'Novel_GRAND_*full*.zip'
append_if_missing ''
append_if_missing '# Transient smoke / probe logs'
append_if_missing 'probe_outputs/check_tags_package_*.log'
append_if_missing 'outputs/slurm_tags_*.out'
append_if_missing 'outputs/slurm_tags_*.err'
append_if_missing ''
append_if_missing '# Large generated artifacts that are not needed on GitHub'
append_if_missing 'outputs/_effective_code_cache/'
append_if_missing 'outputs/*/probe/shards/'
append_if_missing 'outputs/*/train/shards/'
append_if_missing 'outputs/*/eval/shards/'
append_if_missing 'outputs/*/eval/sampled_failures/'
append_if_missing 'outputs/*/models/*.pt'
append_if_missing 'outputs/*/models/*.pth'

echo 'Updated .gitignore with recommended Novel_GRAND ignore patterns.'
