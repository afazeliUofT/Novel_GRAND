#!/bin/bash -l
#SBATCH --job-name=tags-legacy-probe
#SBATCH --account=def-rsadve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=outputs/slurm_tags_probe_%j.out
#SBATCH --error=outputs/slurm_tags_probe_%j.err

set -euo pipefail

cd /home/rsadve1/scratch/Novel_GRAND
mkdir -p outputs
source env/activate_fir.sh

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CFG="${NOVEL_GRAND_CONFIG:-configs/fir_legacy_probe.yaml}"
python -m novel_grand.scripts.probe_legacy --config "${CFG}"
