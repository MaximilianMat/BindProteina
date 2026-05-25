#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --partition=gpu_a100
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --mem 40gb
#SBATCH --time 3:00:00
#SBATCH --output=laproteina_generate_%A.log

# Initialise environment and modules
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/BindProteina
export LD_LIBRARY_PATH=${CONDA_BASE}/lib
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/nfs/homedirs/matscher/miniconda3/envs/BindProteina/lib/python3.11/site-packages/triton/backends/nvidia"

# Get the directory where the script is located
SCRIPT_DIR="/nfs/homedirs/matscher/repos/BindProteina"
cd $SCRIPT_DIR

echo "Running the LaProteina initial generation workflow"
# Pass all arguments directly to the python script
python -u pipeline/generate.py "$@"
