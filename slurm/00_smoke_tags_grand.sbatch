#!/bin/bash -l
#SBATCH --job-name=tags-smoke
#SBATCH --account=def-rsadve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/slurm_tags_smoke_%j.out
#SBATCH --error=outputs/slurm_tags_smoke_%j.err

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

python -m novel_grand.scripts.smoke --config configs/fir_smoke.yaml
