#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --partition=gpu_v100,gpu_a100,gpu_l40s,gpu_h200
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --mem 40gb
#SBATCH --time 08:00:00
#SBATCH --output=reseed_shard_%A_%a.log
#SBATCH --open-mode=append
#SBATCH --requeue
# Send SIGUSR1 to the batch shell 120s before the time limit hits, so we can ask Slurm
# to requeue this exact array task before it gets hard-killed. See the trap below.
#SBATCH --signal=B:USR1@120

# Sharded re-seeding ablation: one array task per strain, one array per round.
#
# Rounds are inherently sequential (round k's champion seeds round k+1), but strains
# are independent, so the parallel axis is the strain. Each array task runs ONE strain
# for ONE round and checkpoints; the next round's array is submitted with a dependency
# on the previous one.
#
# Every shard re-derives the same global founder ordering (selection is deterministic)
# and keeps only its own strain ids, writing into its own output tree.
#
# Usage:
#   sbatch --array=0-19 reseed_shard.slurm \
#          --settings settings_reseed/CD19.json \
#          --base_out /nfs/students/matscher/reseed_output/cd19 \
#          --n_shards 20 --round 1
#
# Any further flags are forwarded to the driver unchanged.
#
# Auto-requeue on time limit: with --requeue and --signal above, Slurm warns this shell
# 120s before the wall-clock hits and this script asks Slurm to requeue the SAME array
# task (same job id, same output paths). --resume (below) then picks up at the last
# CHECKPOINTED round automatically - checkpointing is per-round, so the round that was
# in progress when time ran out is redone from scratch (the driver wipes its own
# leftover gen<k> dir at startup for exactly this reason), never the rounds before it.
# Capped at MAX_RESTARTS so a shard that can never finish one round within the time
# limit fails loudly instead of looping forever unattended.

set -eo pipefail

MAX_RESTARTS=6
RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"
echo "Attempt ${RESTART_COUNT} (max ${MAX_RESTARTS}) for job ${SLURM_JOB_ID:-n/a}, array task ${SLURM_ARRAY_TASK_ID:-n/a}"

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/bin/activate" "${CONDA_BASE}/envs/BindProteina"
export LD_LIBRARY_PATH="${CONDA_BASE}/lib"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/nfs/homedirs/matscher/miniconda3/envs/BindProteina/lib/python3.11/site-packages/triton/backends/nvidia"

SCRIPT_DIR="/nfs/homedirs/matscher/repos/BindProteina"
cd "$SCRIPT_DIR"

SETTINGS=""
BASE_OUT=""
N_SHARDS=""
ROUND=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --settings)  SETTINGS="$2";  shift 2 ;;
        --base_out)  BASE_OUT="$2";  shift 2 ;;
        --n_shards)  N_SHARDS="$2";  shift 2 ;;
        --round)     ROUND="$2";     shift 2 ;;
        *)           EXTRA_ARGS+=("$1"); shift ;;
    esac
done

for v in SETTINGS BASE_OUT N_SHARDS ROUND; do
    if [[ -z "${!v}" ]]; then
        echo "ERROR: --${v,,} is required"
        exit 1
    fi
done

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_OUT="${BASE_OUT}/shard$(printf '%02d' "$SHARD")"

echo "Re-seeding shard ${SHARD}/${N_SHARDS}, round ${ROUND}"
echo "  settings:   ${SETTINGS}"
echo "  output:     ${SHARD_OUT}"

if [[ "$RESTART_COUNT" -ge "$MAX_RESTARTS" ]]; then
    echo "ERROR: hit MAX_RESTARTS=${MAX_RESTARTS} without finishing round ${ROUND}."
    echo "This round is not completing within the time limit even across ${MAX_RESTARTS}"
    echo "attempts - raise --time and resubmit manually rather than requeuing further."
    exit 1
fi

# Requeue this exact array task and get out of the way. Slurm terminates the job
# shortly after scontrol requeue returns, so explicitly killing PID here just avoids a
# few extra seconds of GPU time spent on a round that is about to be wiped and redone.
_on_time_limit() {
    echo "Time limit approaching - requeuing shard ${SHARD} (job ${SLURM_JOB_ID})"
    scontrol requeue "${SLURM_JOB_ID}"
    kill -TERM "$PY_PID" 2>/dev/null || true
    wait "$PY_PID" 2>/dev/null || true
    exit 0
}
trap _on_time_limit USR1

# --resume is always safe: with no checkpoint it starts at round 0 and initialises
# generation 0, otherwise it picks up after the last completed round. Combined with
# --rounds ${ROUND} that means this job runs exactly round ${ROUND}. Backgrounded +
# waited so the USR1 trap above actually interrupts it instead of queuing behind a
# blocking foreground command.
python -u -m pipeline.reseed.reseed_ablation \
    --settings "$SETTINGS" \
    --n_shards "$N_SHARDS" \
    --shard_index "$SHARD" \
    --output_base_dir "$SHARD_OUT" \
    --rounds "$ROUND" \
    --resume \
    "${EXTRA_ARGS[@]}" &
PY_PID=$!
wait "$PY_PID"
