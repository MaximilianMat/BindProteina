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

# Direct-binder inference mode:
# - Choose variant with DIRECT_BINDER_VARIANT=aa|tip (default: aa)
# - Or pass INFERENCE_CONFIG explicitly to override.
DIRECT_BINDER_VARIANT="${DIRECT_BINDER_VARIANT:-aa}"
INFERENCE_CONFIG="${INFERENCE_CONFIG:-}"

if [[ -z "${INFERENCE_CONFIG}" ]]; then
  case "${DIRECT_BINDER_VARIANT}" in
    aa)
      INFERENCE_CONFIG="inference_motif_idx_aa_binder"
      ;;
    tip)
      INFERENCE_CONFIG="inference_motif_idx_tip_binder"
      ;;
    *)
      echo "Unknown DIRECT_BINDER_VARIANT='${DIRECT_BINDER_VARIANT}'. Expected 'aa' or 'tip'."
      exit 1
      ;;
  esac
fi

if [[ ! -f "configs/${INFERENCE_CONFIG}.yaml" ]]; then
  echo "Config not found: configs/${INFERENCE_CONFIG}.yaml"
  exit 1
fi

echo "Running direct-binder inference in env: ${ENV_NAME}"
echo "Using config: ${INFERENCE_CONFIG}"
python proteinfoundation/generate.py --config_name "${INFERENCE_CONFIG}"
