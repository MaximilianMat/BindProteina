#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --partition=gpu_a100
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --mem 40gb
#SBATCH --time 48:00:00
#SBATCH --output=laproteina_evolution_%A.log

# Initialise environment and modules
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/BindCraft
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

# Get the directory where the script is located
SCRIPT_DIR="/nfs/homedirs/matscher/repos/BindCraft"
cd $SCRIPT_DIR

echo "Running the LaProteina evolution workflow"
# Pass all arguments directly to the python script
python -u laproteina/proteinfoundation/evolve.py "$@"