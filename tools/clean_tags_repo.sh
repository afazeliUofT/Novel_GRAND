#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${REPO_DIR}"

echo "Cleaning generated TAGS-GRAND artifacts under ${REPO_DIR}"

echo "- removing Python cache directories"
find . -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete 2>/dev/null || true

echo "- removing editable-install metadata"
rm -rf novel_grand.egg-info

echo "- removing local overlay archives from repo root"
rm -f Novel_GRAND_*overlay.zip Novel_GRAND_*package.zip || true

echo "- removing old TAGS-GRAND outputs and logs"
rm -rf       outputs/fir_tags_grand_default       outputs/fir_tags_grand_autotuned       outputs/fir_tags_grand_smoke       outputs/fir_legacy_probe_default       outputs/_effective_code_cache || true

rm -f outputs/slurm_tags_*.out outputs/slurm_tags_*.err || true
rm -f probe_outputs/check_tags_package_*.log || true

echo "- preserving .venv-fir and environment probe logs"
echo "Done."
