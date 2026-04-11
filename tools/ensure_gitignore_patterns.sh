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
append_if_missing '# Python cache / build noise'
append_if_missing '__pycache__/'
append_if_missing '*.pyc'
append_if_missing '*.pyo'
append_if_missing '*.egg-info/'
append_if_missing 'probe_outputs/check_tags_package_*.log'

echo 'Updated .gitignore with Python cache and editable-install patterns.'
