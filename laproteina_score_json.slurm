#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --partition=gpu_a100,gpu_h100,gpu_h200
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --mem 40gb
#SBATCH --time 72:00:00
#SBATCH --output=laproteina_score_json_%A.log

set -eo pipefail

# Fold and score complexes from a JSON input using Protenix.
#
# Usage:
#   sbatch laproteina_score_json.slurm \
#     --input /path/to/input.json \
#     --output_dir /path/to/output \
#     --fetch_all_msas

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/bin/activate" "${CONDA_BASE}/envs/BindProteina"
export LD_LIBRARY_PATH="${CONDA_BASE}/lib"

SCRIPT_DIR="/nfs/homedirs/matscher/repos/BindProteina"
cd "${SCRIPT_DIR}"

echo "Running scoring pipeline on JSON input"
echo "Arguments: $*"

python -u pipeline/score_json.py "$@"
