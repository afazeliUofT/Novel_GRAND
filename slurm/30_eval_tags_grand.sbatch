#!/bin/bash -l
#SBATCH --job-name=tags-eval
#SBATCH --account=def-rsadve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/slurm_tags_eval_%j.out
#SBATCH --error=outputs/slurm_tags_eval_%j.err

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

CFG="${NOVEL_GRAND_CONFIG:-configs/fir_default.yaml}"
echo "Using config: ${CFG}"

python -m novel_grand.scripts.evaluate --config "${CFG}"
