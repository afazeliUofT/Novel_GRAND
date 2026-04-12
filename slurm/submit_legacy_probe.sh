#!/bin/bash
set -euo pipefail

cd /home/rsadve1/scratch/Novel_GRAND
mkdir -p outputs
CFG="${1:-configs/fir_legacy_probe.yaml}"
if [[ ! -f "${CFG}" ]]; then
  echo "ERROR: config file not found: ${CFG}" >&2
  exit 1
fi
JID=$(sbatch --parsable --export=ALL,NOVEL_GRAND_CONFIG="${CFG}" slurm/05_probe_legacy_tags_grand.sbatch)
echo "Submitted legacy probe job: ${JID}"
