#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --partition=gpu_a100,gpu_h100,gpu_h200
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --mem 40gb
#SBATCH --time 72:00:00
#SBATCH --output=laproteina_score_existing_%A.log

set -eo pipefail

# Score already-generated complexes with AF2 + PyRosetta.
#
# Usage:
#   sbatch laproteina_score_existing.slurm \
#     --input_dir /path/to/pdbs \
#     --output_dir /path/to/scored_output \
#     --filters settings_filters/laProteina_filters.json
#
# Optional:
#   --af2_model 1
#   TARGET_LENGTH=220 sbatch ...   # repairs single-chain outputs by splitting A/B on residue index

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/bin/activate" "${CONDA_BASE}/envs/BindProteina"
export LD_LIBRARY_PATH="${CONDA_BASE}/lib"

SCRIPT_DIR="/nfs/homedirs/matscher/repos/BindProteina"
cd "${SCRIPT_DIR}"

echo "Running scoring-only pipeline on existing PDBs"
echo "Arguments: $*"

INPUT_DIR=""
OUTPUT_DIR=""
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_dir|-i)
      INPUT_DIR="$2"
      ARGS+=("$1" "$2")
      shift 2
      ;;
    --output_dir|-o)
      OUTPUT_DIR="$2"
      ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      ARGS+=("$1")
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

FILTERED_INPUT_DIR="${OUTPUT_DIR%/}/_chainB_only_input"
mkdir -p "${FILTERED_INPUT_DIR}"

echo "Pre-filtering input PDBs to keep only files with chain B..."
python - "${INPUT_DIR}" "${FILTERED_INPUT_DIR}" "${TARGET_LENGTH:-}" <<'PY'
from pathlib import Path
import shutil
import sys

in_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
target_length_raw = sys.argv[3].strip()
target_length = int(target_length_raw) if target_length_raw else None
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

pdbs = sorted(in_dir.rglob("*.pdb"))
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

print(f"Input PDBs: {len(pdbs)}")
print(f"Chain-B PDBs kept (native): {kept_native}")
print(f"Chain-B PDBs kept (repaired): {kept_repaired}")
print(f"Dropped (no chain B): {dropped}")
if kept_native + kept_repaired == 0:
    raise SystemExit(
        "No chain-B PDBs found. "
        "If these are single-chain direct-binder outputs, rerun with TARGET_LENGTH=<target_nres>."
    )
PY

FINAL_ARGS=()
SKIP_NEXT=0
for ((i=0; i<${#ARGS[@]}; i++)); do
  if [[ ${SKIP_NEXT} -eq 1 ]]; then
    SKIP_NEXT=0
    continue
  fi
  if [[ "${ARGS[$i]}" == "--input_dir" || "${ARGS[$i]}" == "-i" ]]; then
    FINAL_ARGS+=("${ARGS[$i]}" "${FILTERED_INPUT_DIR}")
    SKIP_NEXT=1
  else
    FINAL_ARGS+=("${ARGS[$i]}")
  fi
done

python -u pipeline/score_existing_two_step.py "${FINAL_ARGS[@]}"
