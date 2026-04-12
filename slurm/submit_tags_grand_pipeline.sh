#!/bin/bash
set -euo pipefail

cd /home/rsadve1/scratch/Novel_GRAND
mkdir -p outputs

DEFAULT_CFG="configs/fir_default.yaml"
PROBE_CFG="outputs/fir_legacy_probe_default/reports/recommended_full_config.yaml"

if [[ $# -ge 1 && -n "${1}" ]]; then
  CFG="$1"
elif [[ -f "${PROBE_CFG}" ]]; then
  CFG="${PROBE_CFG}"
else
  CFG="${DEFAULT_CFG}"
fi

if [[ ! -f "${CFG}" ]]; then
  echo "ERROR: config file not found: ${CFG}" >&2
  exit 1
fi

EXP_NAME=$(python - <<'PY' "${CFG}"
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8'))
print(cfg.get('experiment_name', 'unknown_experiment'))
PY
)

if [[ "${CFG}" == "${PROBE_CFG}" ]]; then
  CFG_SOURCE="probe_recommendation"
elif [[ "${CFG}" == "${DEFAULT_CFG}" ]]; then
  CFG_SOURCE="default"
else
  CFG_SOURCE="explicit"
fi

python - <<'PY' "${CFG}" "${EXP_NAME}" "${CFG_SOURCE}"
import json, sys
from pathlib import Path
meta = {
    'config_path': sys.argv[1],
    'experiment_name': sys.argv[2],
    'config_source': sys.argv[3],
}
Path('outputs/_last_pipeline_run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
PY

echo "Using config: ${CFG}"
echo "Experiment : ${EXP_NAME}"
EXPORTS="ALL,NOVEL_GRAND_CONFIG=${CFG}"

jid_collect=$(sbatch --parsable --export="${EXPORTS}" slurm/10_collect_train_tags_grand.sbatch)
echo "Submitted collect job: ${jid_collect}"

jid_train=$(sbatch --parsable --export="${EXPORTS}" --dependency=afterok:${jid_collect} slurm/20_train_tags_grand.sbatch)
echo "Submitted train job: ${jid_train}"

jid_eval=$(sbatch --parsable --export="${EXPORTS}" --dependency=afterok:${jid_train} slurm/30_eval_tags_grand.sbatch)
echo "Submitted eval job: ${jid_eval}"

jid_report=$(sbatch --parsable --export="${EXPORTS}" --dependency=afterok:${jid_eval} slurm/40_report_tags_grand.sbatch)
echo "Submitted report job: ${jid_report}"
