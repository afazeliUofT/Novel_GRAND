#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${REPO_DIR}"

check_file() {
  local p="$1"
  if [[ -f "${p}" ]]; then
    echo "[OK] ${p}"
  else
    echo "[MISSING] ${p}"
  fi
}

check_dir() {
  local p="$1"
  if [[ -d "${p}" ]]; then
    echo "[OK] ${p}/"
  else
    echo "[MISSING] ${p}/"
  fi
}

echo "=== Repo status ==="
check_file "outputs/fir_tags_grand_smoke/smoke/smoke_summary.json"
check_file "outputs/fir_legacy_probe_default/reports/legacy_probe_summary.csv"
check_file "outputs/fir_legacy_probe_default/reports/recommended_ebn0.json"
check_file "outputs/fir_legacy_probe_default/reports/recommended_full_config.yaml"
check_dir  "outputs/fir_tags_grand_autotuned/train"
check_dir  "outputs/fir_tags_grand_autotuned/models"
check_dir  "outputs/fir_tags_grand_autotuned/eval"
check_dir  "outputs/fir_tags_grand_autotuned/reports"
check_file "outputs/fir_tags_grand_autotuned/reports/summary_eval.csv"
check_file "outputs/fir_tags_grand_autotuned/reports/summary.md"
check_file "outputs/fir_tags_grand_autotuned/reports/worker_diversity_summary.csv"
check_file "outputs/fir_tags_grand_autotuned/reports/tags_stage_contribution_summary.csv"
check_file "outputs/fir_tags_grand_autotuned/reports/tags_stage_visit_summary.csv"
check_file "outputs/fir_tags_grand_autotuned/reports/gap_to_oracle_summary.csv"
check_file "outputs/fir_tags_grand_autotuned/reports/net_exact_success_rate_vs_ebn0.png"

if [[ -f outputs/fir_tags_grand_autotuned/reports/summary.md && -f outputs/fir_tags_grand_autotuned/reports/summary_eval.csv ]]; then
  if [[ outputs/fir_tags_grand_autotuned/reports/summary.md -ot outputs/fir_tags_grand_autotuned/reports/summary_eval.csv ]]; then
    echo
    echo "[WARNING] summary.md is older than summary_eval.csv"
    echo "          The report stage may have failed after writing CSV files."
  fi
fi

nested=$(find outputs -type d -path '*/outputs/fir_tags_grand_autotuned' 2>/dev/null | head -n 1 || true)
if [[ -n "${nested}" ]]; then
  echo
  echo "[WARNING] Accidental nested autotuned output detected: ${nested}"
  echo "          This usually means repo_root was inferred from the config path instead of the actual repo root."
fi

if [[ -f outputs/fir_legacy_probe_default/reports/recommended_ebn0.json && -f outputs/fir_legacy_probe_default/reports/recommended_full_config.yaml ]]; then
  python - <<'PY'
from pathlib import Path
import json
import yaml
root = Path('outputs/fir_legacy_probe_default/reports')
js = json.loads((root / 'recommended_ebn0.json').read_text())
yml = yaml.safe_load((root / 'recommended_full_config.yaml').read_text())
rec_json = list(map(float, js.get('recommended_ebn0_db_list', [])))
rec_yaml = list(map(float, yml.get('simulation', {}).get('ebn0_db_list', [])))
if rec_json != rec_yaml:
    print('\n[WARNING] Probe recommendation mismatch detected')
    print(f'          recommended_ebn0.json      = {rec_json}')
    print(f'          recommended_full_config.yaml = {rec_yaml}')
PY
fi
