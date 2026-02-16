#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --partition=gpu_a100,gpu_h100,gpu_h200
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --mem 48G
#SBATCH --time 48:00:00
#SBATCH --output=laproteina_ft_main_%A.log

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

# Avoid cross-device link when saving checkpoints: use same filesystem as checkpoint dir
export TMPDIR="/nfs/students/matscher/laproteina_checkpoints/.tmp"
mkdir -p "$TMPDIR"

echo "Running binder FT main phase in env: ${ENV_NAME}"
bash script_utils/binder_ft_stage_schedule.sh main
