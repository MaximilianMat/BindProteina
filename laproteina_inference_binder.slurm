#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --partition=gpu_a100,gpu_h100,gpu_h200
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --mem 48G
#SBATCH --time 4:00:00
#SBATCH --output=laproteina_inference_binder_%A.log

set -eo pipefail

ENV_NAME=BindProteina
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE=$(conda info --base)
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "Could not locate conda initialization script."
  exit 1
fi
conda activate "${ENV_NAME}"

SCRIPT_DIR="/nfs/homedirs/matscher/repos/BindProteina"
LAPROTEINA_ROOT="${SCRIPT_DIR}/laproteina"
cd "$LAPROTEINA_ROOT"

echo "Running binder inference in env: ${ENV_NAME}"
python proteinfoundation/generate.py --config_name inference_motif_idx_aa_binder
