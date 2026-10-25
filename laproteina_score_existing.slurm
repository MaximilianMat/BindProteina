#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --partition=gpu_v100,gpu_l40s,gpu_a100
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --mem 40gb
#SBATCH --time 04:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=laproteina_score_existing_%A_%a.log

set -eo pipefail

# Score already-generated complexes with AF2 + PyRosetta.
#
# Two independent ways to cut wall-clock time, and they compose:
#
# 1. Short jobs that requeue themselves instead of one long job. --time is
#    deliberately short (03:30:00) so the job backfills fast; SIGUSR1 fires
#    120s before the limit, the python child is stopped cleanly, and the job
#    requeues itself (scontrol requeue, or a fresh sbatch if that's disallowed)
#    to keep going from where score_complexes()'s own CSV-resume logic left
#    off. This alone fixes "resubmitting is slow" only in the sense of "it
#    happens automatically" -- it is still one worker at a time.
#
# 2. Parallel workers via a SLURM array: submit as an array and each task
#    scores a disjoint 1/N slice of the input into its own shard directory,
#    so N workers actually run concurrently. This is what cuts wall-clock
#    time by ~N.
#
# Usage (10 parallel workers, at most 10 running at once, each self-requeuing
# on timeout until its shard is done):
#   sbatch --array=0-9%10 laproteina_score_existing.slurm \
#     --input_dir /path/to/pdbs \
#     --output_dir /path/to/scored_output \
#     --filters settings_filters/laProteina_filters.json
#
#   # once the array is done (fast, fine on the login node):
#   python scripts/merge_score_shards.py --output_dir /path/to/scored_output
#
# Without --array it behaves as before (one worker, output straight into
# --output_dir), just with self-requeue instead of a single long job.
#
# The array must be a contiguous range starting at 0 -- task id == shard
# index. To requeue one failed shard on its own after the array job is gone,
# pass the original shard count explicitly (and pin the AF2 model -- printed
# in that shard's log -- so the requeued shard matches the others):
#   NUM_SHARDS=10 sbatch --array=3 laproteina_score_existing.slurm \
#     --af2_model <N> ... (same --output_dir)
#
# Optional:
#   --folding_model protenix --protenix_model protenix_base_20250630_v1.0.0
#   --af2_model 1                  # af2 path only, see the pinning note below
#   TARGET_LENGTH=220 sbatch ...   # repairs single-chain outputs by splitting A/B on residue index
#
# Note: under --folding_model protenix each shard generates the target MSA
# itself (~10s per shard against the online server) -- duplicated work, but
# small enough not to be worth sharing between shards.

ORIGINAL_ARGS=("$@")

# Signal handler for Slurm timeout (SIGUSR1 sent 120s before walltime limit)
handle_timeout() {
  echo "========================================================"
  echo "Slurm walltime limit approaching (SIGUSR1 received)."
  echo "Terminating scoring process cleanly..."
  echo "========================================================"
  if [[ -n "${PY_PID:-}" ]] && kill -0 "${PY_PID}" 2>/dev/null; then
    kill -TERM "${PY_PID}"
    wait "${PY_PID}" 2>/dev/null || true
  fi

  echo "Requeuing job..."
  if ! scontrol requeue "${SLURM_JOB_ID:-}" 2>/dev/null; then
    echo "scontrol requeue failed or not allowed; submitting new job via sbatch..."
    if [[ "${NUM_SHARDS:-1}" -gt 1 ]]; then
      # Preserve shard identity across a fresh sbatch: resubmit as a
      # single-task array at this exact index, carrying NUM_SHARDS along as
      # an exported env var so the shard math resolves the same way.
      sbatch --array="${TASK_ID}" --export=ALL,NUM_SHARDS="${NUM_SHARDS}" "$0" "${ORIGINAL_ARGS[@]}"
    else
      sbatch "$0" "${ORIGINAL_ARGS[@]}"
    fi
  fi
  exit 0
}

trap 'handle_timeout' USR1

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/bin/activate" "${CONDA_BASE}/envs/BindProteina"
export LD_LIBRARY_PATH="${CONDA_BASE}/lib"

SCRIPT_DIR="/nfs/homedirs/matscher/repos/BindProteina"
cd "${SCRIPT_DIR}"

echo "Running scoring-only pipeline on existing PDBs"
echo "Arguments: $*"

INPUT_DIR=""
OUTPUT_DIR=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_dir|-i)
      INPUT_DIR="$2"
      shift 2
      ;;
    --output_dir|-o)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${INPUT_DIR}" ]]; then
  echo "Error: --input_dir is required"
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "Error: --output_dir is required"
  exit 1
fi

# ── Shard resolution ──────────────────────────────────────────────
# Outside an array job this collapses to shard 0 of 1, i.e. the old behaviour.
# NUM_SHARDS can be set at submit time to override the derived count; that is
# what makes requeuing one task of a finished array (--array=3) score shard 3
# only, instead of treating itself as a one-shard run over the whole input.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ -z "${NUM_SHARDS}" ]]; then
  if [[ -n "${SLURM_ARRAY_TASK_COUNT}" ]]; then
    NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT}"
  elif [[ -n "${SLURM_ARRAY_TASK_MAX}" ]]; then
    NUM_SHARDS=$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))
  else
    NUM_SHARDS=1
  fi
fi
if [[ "${TASK_ID}" -ge "${NUM_SHARDS}" ]]; then
  echo "Error: array task ${TASK_ID} is outside the shard range 0..$((NUM_SHARDS - 1)). "
  echo "Use a contiguous array starting at 0, or set NUM_SHARDS at submit time."
  exit 1
fi

if [[ "${NUM_SHARDS}" -gt 1 ]]; then
  WORK_DIR="${OUTPUT_DIR%/}/shard_$(printf '%03d' "${TASK_ID}")"
  echo "Array task ${TASK_ID} of ${NUM_SHARDS} -> ${WORK_DIR}"
else
  WORK_DIR="${OUTPUT_DIR%/}"
fi

# Which folding model will actually run (score_existing_two_step.py defaults to
# af2). Only used to decide whether the AF2-model pinning below applies.
FOLDING_MODEL="af2"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  case "${EXTRA_ARGS[$i]}" in
    --folding_model)
      FOLDING_MODEL="${EXTRA_ARGS[$((i + 1))]}"
      ;;
    --folding_model=*)
      FOLDING_MODEL="${EXTRA_ARGS[$i]#*=}"
      ;;
  esac
done
echo "Folding model: ${FOLDING_MODEL}"

# On the AF2 path all shards must refold with the same model, otherwise the
# merged table mixes models: score_existing_two_step.py draws a random model per
# process, so in array mode we derive one deterministically from the shared
# array job id (still random per submission, identical across its tasks).
#
# Only on the AF2 path. --af2_model is not just the refolding model index -- it
# also selects which "<N>_<metric>" thresholds are read from the filter JSON
# (scorer.py: af2_model_num_for_filter = af2_model_idx or 1), so injecting it
# under --folding_model protenix would silently shift filter selection for a
# model that never ran.
if [[ "${FOLDING_MODEL}" == "af2" && "${NUM_SHARDS}" -gt 1 && " ${EXTRA_ARGS[*]} " != *" --af2_model "* ]]; then
  AF2_MODEL=$(( (${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}} % 5) + 1 ))
  EXTRA_ARGS+=("--af2_model" "${AF2_MODEL}")
  echo "Pinning AF2 model ${AF2_MODEL} for all shards of array ${SLURM_ARRAY_JOB_ID:-none}"
  echo "(when requeuing a single shard later, pass --af2_model ${AF2_MODEL} to match)"
fi

FILTERED_INPUT_DIR="${WORK_DIR}/_chainB_only_input"
mkdir -p "${FILTERED_INPUT_DIR}"

echo "Pre-filtering input PDBs to keep only files with chain B..."
python - "${INPUT_DIR}" "${FILTERED_INPUT_DIR}" "${TARGET_LENGTH:-}" "${TASK_ID}" "${NUM_SHARDS}" <<'PY'
from pathlib import Path
import shutil
import sys

in_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
target_length_raw = sys.argv[3].strip()
target_length = int(target_length_raw) if target_length_raw else None
task_id = int(sys.argv[4])
num_shards = int(sys.argv[5])
out_dir.mkdir(parents=True, exist_ok=True)

def _has_chain_b(lines):
    for line in lines:
        if line.startswith("ATOM") and len(line) > 21 and line[21] == "B":
            return True
    return False

def _repair_single_chain_by_target_len(lines, tgt_len):
    """Relabel residues <= tgt_len to chain A and > tgt_len to chain B."""
    if tgt_len is None:
        return None
    repaired = []
    saw_b = False
    for line in lines:
        if line.startswith("ATOM") and len(line) > 26:
            try:
                res_id = int(line[22:26].strip())
            except ValueError:
                repaired.append(line)
                continue
            new_chain = "A" if res_id <= tgt_len else "B"
            if new_chain == "B":
                saw_b = True
            # PDB chain id is column 22 (0-based index 21)
            line = f"{line[:21]}{new_chain}{line[22:]}"
        repaired.append(line)
    if not saw_b:
        return None
    return repaired

all_pdbs = sorted(in_dir.rglob("*.pdb"))
# Strided slice: deterministic, disjoint across shards, and independent of how
# many files each shard actually keeps.
pdbs = all_pdbs[task_id::num_shards]
kept_native = 0
kept_repaired = 0
dropped = 0
for idx, pdb in enumerate(pdbs):
    lines = pdb.read_text(errors="ignore").splitlines(keepends=True)
    if _has_chain_b(lines):
        shutil.copy2(pdb, out_dir / pdb.name)
        kept_native += 1
        continue
    repaired = _repair_single_chain_by_target_len(lines, target_length)
    if repaired is not None:
        (out_dir / pdb.name).write_text("".join(repaired))
        kept_repaired += 1
    else:
        dropped += 1

print(f"Input PDBs: {len(all_pdbs)} (this shard: {len(pdbs)})")
print(f"Chain-B PDBs kept (native): {kept_native}")
print(f"Chain-B PDBs kept (repaired): {kept_repaired}")
print(f"Dropped (no chain B): {dropped}")
if kept_native + kept_repaired == 0:
    if not pdbs:
        print("Empty shard (fewer input PDBs than array tasks).")
        raise SystemExit(0)
    raise SystemExit(
        "No chain-B PDBs found. "
        "If these are single-chain direct-binder outputs, rerun with TARGET_LENGTH=<target_nres>."
    )
PY

if [[ $(find "${FILTERED_INPUT_DIR}" -maxdepth 1 -name "*.pdb" | wc -l) -eq 0 ]]; then
  echo "Nothing to score in this shard, exiting."
  exit 0
fi

python -u pipeline/score_existing_two_step.py \
  --input_dir "${FILTERED_INPUT_DIR}" \
  --output_dir "${WORK_DIR}" \
  "${EXTRA_ARGS[@]}" &
PY_PID=$!
wait "${PY_PID}"
