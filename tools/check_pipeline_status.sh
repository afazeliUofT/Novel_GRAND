#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${REPO_DIR}"

check_file() {
  local p="$1"
  if [[ -f "${p}" ]]; then
    echo "[OK]      ${p}"
  else
    echo "[MISSING] ${p}"
  fi
}

check_dir() {
  local p="$1"
  if [[ -d "${p}" ]]; then
    echo "[OK]      ${p}/"
  else
    echo "[MISSING] ${p}/"
  fi
}

echo "=== Repo status ==="
check_file "outputs/fir_tags_grand_smoke/smoke/smoke_summary.json"
check_file "outputs/fir_legacy_probe_default/reports/legacy_probe_summary.csv"
check_file "outputs/fir_legacy_probe_default/reports/recommended_full_config.yaml"

check_dir "outputs/fir_tags_grand_autotuned/train"
check_dir "outputs/fir_tags_grand_autotuned/models"
check_dir "outputs/fir_tags_grand_autotuned/eval"
check_dir "outputs/fir_tags_grand_autotuned/reports"

check_file "outputs/fir_tags_grand_autotuned/reports/summary_eval.csv"
check_file "outputs/fir_tags_grand_autotuned/reports/summary.md"
check_file "outputs/fir_tags_grand_autotuned/reports/net_exact_success_rate_vs_ebn0.png"
