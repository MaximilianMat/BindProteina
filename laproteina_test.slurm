#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --partition=gpu_a100
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --mem 40gb
#SBATCH --time 1:00:00
#SBATCH --output=laproteina_scoring_%A.log

# Initialise environment and modules
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/BindCraft
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

# Get the directory where the script is located
SCRIPT_DIR="/nfs/homedirs/matscher/repos/BindCraft"

echo "Running the LaProteina scoring pipeline"
# Pass all arguments directly to the python module
python test_interface_residues.py evolution_output/evolution_20251228_1637/round_1/scored/af2_predictions/R1B60599_sample_000_model1.pdb
