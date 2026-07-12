#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --partition=gpu_a100
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --mem 40gb
#SBATCH --time 48:00:00
#SBATCH --output=laproteina_batch_%A_%a.log

# Initialise environment and modules
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/BindProteina
export LD_LIBRARY_PATH=${CONDA_BASE}/lib
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/nfs/homedirs/matscher/miniconda3/envs/BindProteina/lib/python3.11/site-packages/triton/backends/nvidia"

SCRIPT_DIR="/nfs/homedirs/matscher/repos/BindProteina"
cd $SCRIPT_DIR

# Parse --batch <manifest_file>; forward any remaining args to generate.py
BATCH_FILE=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch)
            BATCH_FILE="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$BATCH_FILE" ]]; then
    echo "ERROR: --batch <manifest_file> is required"
    exit 1
fi

# SLURM_ARRAY_TASK_ID is 0-based; manifest lines are 1-based
SETTINGS_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$BATCH_FILE")

if [[ -z "$SETTINGS_FILE" ]]; then
    echo "ERROR: no entry at index ${SLURM_ARRAY_TASK_ID} in ${BATCH_FILE}"
    exit 1
fi

echo "Array task ${SLURM_ARRAY_TASK_ID}: running ${SETTINGS_FILE}"
python -u pipeline/generate.py --settings "$SETTINGS_FILE" "${EXTRA_ARGS[@]}"
