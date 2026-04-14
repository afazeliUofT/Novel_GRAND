#!/bin/bash -l
#SBATCH --job-name=tags-train
#SBATCH --account=def-rsadve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/slurm_tags_fit_%j.out
#SBATCH --error=outputs/slurm_tags_fit_%j.err

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

python -m novel_grand.scripts.train_tags --config "${CFG}"
