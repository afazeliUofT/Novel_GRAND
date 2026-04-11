#!/bin/bash
set -euo pipefail

cd /home/rsadve1/scratch/Novel_GRAND
CFG="${1:-configs/fir_legacy_probe.yaml}"
JID=$(sbatch --parsable --export=ALL,NOVEL_GRAND_CONFIG="${CFG}" slurm/05_probe_legacy_tags_grand.sbatch)
echo "Submitted legacy probe job: ${JID}"
