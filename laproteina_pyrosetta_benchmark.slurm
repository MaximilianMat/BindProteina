#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --partition=gpu_a100,gpu_h100
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --mem 40gb
#SBATCH --time 1:00:00
#SBATCH --output=laproteina_pyrosetta_benchmark_%A.log

# PyRosetta Multithreading Benchmark
# Benchmarks PyRosetta relaxation and scoring performance with single vs multithreaded configs
#
# Usage:
#   sbatch laproteina_pyrosetta_benchmark.slurm --pdb path/to/test.pdb --iterations 3
#   sbatch laproteina_pyrosetta_benchmark.slurm --pdb path/to/test.pdb --threads 8 --iterations 5

# Initialise environment and modules
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/BindProteina
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

# Get the directory where the script is located
SCRIPT_DIR="/nfs/homedirs/matscher/repos/BindProteina"
cd $SCRIPT_DIR

echo "Running PyRosetta Multithreading Benchmark"
echo "Arguments: $@"
echo "CPU cores available: $SLURM_CPUS_PER_TASK"
# Pass all arguments directly to the python script
python -u tests/test_pyrosetta_multithreading.py "$@"
