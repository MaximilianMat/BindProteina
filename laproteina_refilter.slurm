#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --partition=cpu_all
#SBATCH --qos=default
#SBATCH --time 1:00:00
#SBATCH --output=laproteina_refilter_%A.log

set -eo pipefail

# Re-apply the current filters to an existing run folder, in place.
# The folding stage is NEVER re-run - metrics are read from the existing
# *_prescoring_results.csv.  PyRosetta runs only for designs that newly pass
# the folding filter and have never been scored.
#
# Usage:
#   # 1. See what would change (writes nothing):
#   sbatch laproteina_refilter.slurm --run_dir /path/to/initial_20260723_0505 --dry_run
#
#   # 2. Apply it:
#   sbatch laproteina_refilter.slurm --run_dir /path/to/initial_20260723_0505
#
# Any other refilter_run.py flag is passed straight through, e.g.
#   --filters settings_filters/laProteina_filters.json
#   --hotspot_residues "14,16,20,64,163,176,178,221,265,271"
#   --retry_failed
#
# Overriding the GPU request (SBATCH CLI args win over the directives above):
#   sbatch --partition=gpu_a100 laproteina_refilter.slurm --run_dir ...
#
# NOTE: this job does not actually need a GPU - the folding stage is skipped and
# PyRosetta is CPU-only.  The gtx1080 request is kept because it was asked for
# and it queues fast; drop --gres/--partition for a CPU partition if you prefer.
#
# Batch over many run folders:
#   for d in /nfs/students/matscher/samples/seed_search_cd19/*/initial_*; do
#     sbatch laproteina_refilter.slurm --run_dir "$d"
#   done

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/bin/activate" "${CONDA_BASE}/envs/BindProteina"
export LD_LIBRARY_PATH="${CONDA_BASE}/lib"

SCRIPT_DIR="/nfs/homedirs/matscher/repos/BindProteina"
cd "${SCRIPT_DIR}"

if [[ $# -eq 0 ]]; then
  echo "Error: --run_dir is required"
  echo "Usage: sbatch laproteina_refilter.slurm --run_dir /path/to/run [--dry_run] [...]"
  exit 1
fi

echo "Re-applying filters to an existing run folder (no refolding)"
echo "Arguments: $*"

python -u pipeline/refilter_run.py "$@"
