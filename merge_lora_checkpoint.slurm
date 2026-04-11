#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --partition=gpu_a100
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --mem 48G
#SBATCH --time 1:00:00
#SBATCH --output=merge_lora_checkpoint_%A.log

set -eo pipefail

CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/BindProteina
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

INPUT_CKPT="$1"
OUTPUT_CKPT="$2"
shift 2

if [[ ! -f "${INPUT_CKPT}" ]]; then
  echo "Input checkpoint not found: ${INPUT_CKPT}"
  exit 1
fi

SCRIPT_DIR="/nfs/homedirs/matscher/repos/BindProteina"
LAPROTEINA_ROOT="${SCRIPT_DIR}/laproteina"
cd "$LAPROTEINA_ROOT"

python proteinfoundation/merge_lora_checkpoint.py \
  --input_ckpt "${INPUT_CKPT}" \
  --output_ckpt "${OUTPUT_CKPT}" \
  "$@"

echo "Merged LoRA checkpoint written to: ${OUTPUT_CKPT}"
