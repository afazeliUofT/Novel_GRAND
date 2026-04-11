#!/bin/bash
set -euo pipefail

cd /home/rsadve1/scratch/Novel_GRAND
mkdir -p outputs

CFG="${1:-configs/fir_default.yaml}"
EXPORTS="ALL,NOVEL_GRAND_CONFIG=${CFG}"

jid_collect=$(sbatch --parsable --export="${EXPORTS}" slurm/10_collect_train_tags_grand.sbatch)
echo "Submitted collect job: ${jid_collect}"

jid_train=$(sbatch --parsable --export="${EXPORTS}" --dependency=afterok:${jid_collect} slurm/20_train_tags_grand.sbatch)
echo "Submitted train job: ${jid_train}"

jid_eval=$(sbatch --parsable --export="${EXPORTS}" --dependency=afterok:${jid_train} slurm/30_eval_tags_grand.sbatch)
echo "Submitted eval job: ${jid_eval}"

jid_report=$(sbatch --parsable --export="${EXPORTS}" --dependency=afterok:${jid_eval} slurm/40_report_tags_grand.sbatch)
echo "Submitted report job: ${jid_report}"
