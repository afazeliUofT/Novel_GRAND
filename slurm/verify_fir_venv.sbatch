#!/bin/bash -l
#SBATCH --job-name=novelgrand-venvchk
#SBATCH --account=def-rsadve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=probe_outputs/venv_verify_%j.out
#SBATCH --error=probe_outputs/venv_verify_%j.err

set -euo pipefail

cd /home/rsadve1/scratch/Novel_GRAND
source env/activate_fir.sh

python tools/verify_sionna_env.py | tee "probe_outputs/venv_verify_compute_${SLURM_JOB_ID}.txt"

python -m pip show \
  torch sionna-no-rt numpy scipy h5py matplotlib importlib-resources \
  pandas PyYAML tqdm psutil | tee "probe_outputs/pip_show_compute_${SLURM_JOB_ID}.txt"
