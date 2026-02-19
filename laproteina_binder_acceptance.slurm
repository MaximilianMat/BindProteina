#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --partition=gpu_a100,gpu_h100,gpu_h200
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --mem 48G
#SBATCH --time 08:00:00
#SBATCH --output=laproteina_binder_acceptance_%A.log

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
cd "${LAPROTEINA_ROOT}"

TASK_NAME="${TASK_NAME:-TARGET_BINDER_AA}"
NSAMPLES="${NSAMPLES:-8}"
RUN_BASELINE="${RUN_BASELINE:-0}"
BASELINE_CKPT_PATH="${BASELINE_CKPT_PATH:-/nfs/students/matscher/laproteina_checkpoints}"
BASELINE_CKPT_NAME="${BASELINE_CKPT_NAME:-LD4_motif_idx_aa.ckpt}"

TMP_OUT="./inference/inference_motif_idx_aa_binder_${TASK_NAME}"
FINETUNED_OUT="./inference/acceptance_finetuned_${TASK_NAME}"
BASELINE_OUT="./inference/acceptance_baseline_${TASK_NAME}"

echo "Running binder acceptance inference in env: ${ENV_NAME}"
echo "TASK_NAME=${TASK_NAME} NSAMPLES=${NSAMPLES} RUN_BASELINE=${RUN_BASELINE}"
echo "NOTE: generate.py in this repo does not accept Hydra-style CLI overrides."
echo "Using task/nsamples from config files."

rm -rf "${TMP_OUT}" "${FINETUNED_OUT}"

echo "=== Generating finetuned samples (config checkpoint) ==="
python proteinfoundation/generate.py \
  --config_name inference_motif_idx_aa_binder
mv "${TMP_OUT}" "${FINETUNED_OUT}"

if [[ "${RUN_BASELINE}" == "1" ]]; then
  rm -rf "${TMP_OUT}" "${BASELINE_OUT}"
  echo "=== Generating baseline LD4 samples ==="
  python proteinfoundation/generate.py \
    --config_name inference_motif_idx_aa_binder_ld4
  mv "${TMP_OUT}" "${BASELINE_OUT}"
fi

echo "=== Running structure diagnostics ==="
if [[ "${RUN_BASELINE}" == "1" ]]; then
  python script_utils/analyze_binder_outputs.py \
    --label baseline \
    --root "${BASELINE_OUT}" \
    --label finetuned \
    --root "${FINETUNED_OUT}"
else
  python script_utils/analyze_binder_outputs.py \
    --label finetuned \
    --root "${FINETUNED_OUT}"
fi

echo "Done."
echo "Finetuned outputs: ${FINETUNED_OUT}"
if [[ "${RUN_BASELINE}" == "1" ]]; then
  echo "Baseline outputs:  ${BASELINE_OUT}"
fi
