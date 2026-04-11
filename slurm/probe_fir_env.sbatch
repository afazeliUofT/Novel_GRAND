#!/bin/bash -l
#SBATCH --job-name=novelgrand-probe
#SBATCH --account=def-rsadve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=probe_outputs/slurm_probe_%j.out
#SBATCH --error=probe_outputs/slurm_probe_%j.err

cd /home/rsadve1/scratch/Novel_GRAND
bash -l tools/fir_env_probe.sh | tee probe_outputs/compute_probe.txt
