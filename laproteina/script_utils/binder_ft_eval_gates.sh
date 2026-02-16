#!/usr/bin/env bash
set -euo pipefail

# Validation gates for binder fine-tuning checkpoints.
# Usage:
#   bash laproteina/script_utils/binder_ft_eval_gates.sh <ckpt_name>
# Example:
#   bash laproteina/script_utils/binder_ft_eval_gates.sh last.ckpt

CKPT_NAME="${1:-last.ckpt}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAPROTEINA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${LAPROTEINA_ROOT}"

python proteinfoundation/generate.py \
  --config_name inference_motif_idx_aa_binder \
  generation.dataset.nsamples=8 \
  ckpt_name="${CKPT_NAME}"

python proteinfoundation/evaluate.py \
  --config_name inference_motif_idx_aa_binder \
  ckpt_name="${CKPT_NAME}"

cat <<'EOF'
Gate checklist:
  [ ] Target-conditioned motif RMSD remains low and stable.
  [ ] Generated binder region is variable across samples.
  [ ] No severe geometry/clash regressions in evaluate outputs.
  [ ] Optional: run downstream AF2-multimer + PyRosetta interface scoring in BindProteina.
EOF
