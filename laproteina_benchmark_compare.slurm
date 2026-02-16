#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --partition=gpu_a100,gpu_h100
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --mem 48gb
#SBATCH --time 12:00:00
#SBATCH --output=laproteina_benchmark_compare_%A.log

# Baseline-vs-finetuned generation benchmark (no scoring).
# Usage:
#   sbatch laproteina_benchmark_compare.slurm /path/to/finetuned.ckpt
# Optional env override:
#   sbatch --export=ENV_NAME=laproteina_env laproteina_benchmark_compare.slurm /path/to/finetuned.ckpt

if [[ -z "${1:-}" ]]; then
  echo "Usage: sbatch laproteina_benchmark_compare.slurm /path/to/finetuned.ckpt"
  exit 1
fi
FT_CKPT="$1"

set -eo pipefail

ENV_NAME=${ENV_NAME:-laproteina_env}
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
cd "$SCRIPT_DIR"

echo "Running baseline-vs-finetuned generation benchmark in env: ${ENV_NAME}"
echo "Fine-tuned checkpoint: ${FT_CKPT}"
bash laproteina/script_utils/binder_benchmark_compare.sh "${FT_CKPT}"
