#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --partition=gpu_a100,gpu_h100
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --mem 40gb
#SBATCH --time 72:00:00
#SBATCH --output=laproteina_evolution_v2_%A.log

# LaProteina V2 Lineage Evolution Script
# Evolves each seed independently with 1:1 best-variant selection per round
#
# Usage (with V2 settings):
#   sbatch laproteina_evolution_v2.slurm --settings settings_evolution/my_target_v2.json
#
# Usage (with existing evolution settings - fully compatible):
#   sbatch laproteina_evolution_v2.slurm --settings settings_evolution/CLD18.json
#
# To resume from checkpoints:
#   sbatch laproteina_evolution_v2.slurm --settings settings_evolution/CLD18.json --resume

# Initialise environment and modules
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/BindProteina
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

# Get the directory where the script is located
SCRIPT_DIR="/nfs/homedirs/matscher/repos/BindProteina"
cd $SCRIPT_DIR

echo "Running LaProteina V2 Lineage Evolution"
echo "Arguments: $@"
# Pass all arguments directly to the python script
python -u pipeline/evolve_v2.py "$@"

