#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --partition=gpu_a100,gpu_h100,gpu_h200
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --mem 60gb
#SBATCH --time 72:00:00
#SBATCH --output=reseed_ablation_%A.log

# Initialise environment and modules
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/BindProteina
export LD_LIBRARY_PATH=${CONDA_BASE}/lib
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/nfs/homedirs/matscher/miniconda3/envs/BindProteina/lib/python3.11/site-packages/triton/backends/nvidia"

SCRIPT_DIR="/nfs/homedirs/matscher/repos/BindProteina"
cd $SCRIPT_DIR

# Usage:
#   sbatch reseed_ablation.slurm --settings settings_reseed/NPT2B.json
#   sbatch reseed_ablation.slurm --settings settings_reseed/NPT2B.json --resume
#   # small pilot:
#   sbatch reseed_ablation.slurm --settings settings_reseed/NPT2B.json --n_strains 1 --samples_per_cut 2 --rounds 1
echo "Running the re-seeding ablation driver"
python -u -m pipeline.reseed.reseed_ablation "$@"
