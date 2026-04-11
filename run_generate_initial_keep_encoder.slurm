#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --partition=gpu_a100
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --mem 48G
#SBATCH --time 1:00:00
#SBATCH --output=run_generate_initial_keep_encoder_%A.log

set -euo pipefail

# Usage:
#   sbatch script_utils/run_generate_initial_keep_encoder.slurm.sh -- \
#     /path/to/settings.yaml [generate_initial args...]

REPO_ROOT="${REPO_ROOT:-/nfs/homedirs/matscher/repos/BindProteina}"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-${REPO_ROOT}/scripts/run_generate_initial_keep_encoder.py}"
GENERATE_SCRIPT="${GENERATE_SCRIPT:-${REPO_ROOT}/laproteina/proteinfoundation/generate_initial.py}"

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "REPO_ROOT does not exist: ${REPO_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${LAUNCH_SCRIPT}" ]]; then
  LAUNCH_SCRIPT="${REPO_ROOT}${LAUNCH_SCRIPT}"
fi
if [[ ! -f "${GENERATE_SCRIPT}" ]]; then
  if [[ -f "${REPO_ROOT}/proteinfoundation/generate_initial.py" ]]; then
    GENERATE_SCRIPT="${REPO_ROOT}/proteinfoundation/generate_initial.py"
  elif [[ -f "${REPO_ROOT}/generate_initial.py" ]]; then
    GENERATE_SCRIPT="${REPO_ROOT}/generate_initial.py"
  else
    echo "Could not locate generate_initial.py under ${REPO_ROOT}" >&2
    exit 1
  fi
fi

if [[ ! -f "${LAUNCH_SCRIPT}" ]]; then
  echo "Could not locate run_generate_initial_keep_encoder.py. Checked:"
  echo "  ${LAUNCH_SCRIPT}"
  echo "  ${REPO_ROOT}/scripts/run_generate_initial_keep_encoder.py"
  exit 1
fi

mkdir -p "${REPO_ROOT}/slurm_out"
mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

if [[ $# -eq 0 ]]; then
  echo "Usage: sbatch run_generate_initial_keep_encoder.slurm.sh -- <generate_initial args...>" >&2
  exit 1
fi

python "${LAUNCH_SCRIPT}" \
  --repo-root "${REPO_ROOT}" \
  --generate-script "${GENERATE_SCRIPT}" \
  -- "$@"
