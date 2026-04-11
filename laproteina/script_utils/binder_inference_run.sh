#!/usr/bin/env bash
set -euo pipefail

# Base inference run: sample binders with LD4 checkpoint.
# Output: ./inference/inference_motif_idx_aa_binder_TARGET_BINDER_AA/
#
# Usage:
#   bash laproteina/script_utils/binder_inference_run.sh
#
# Override checkpoint (e.g. fine-tuned):
#   bash laproteina/script_utils/binder_inference_run.sh /path/to/finetuned.ckpt
#
# Convert LoRA checkpoint before inference (no code changes required):
#   python proteinfoundation/merge_lora_checkpoint.py \
#     --input_ckpt /path/to/finetuned_lora.ckpt \
#     --output_ckpt /path/to/finetuned_lora_merged.ckpt
#   bash laproteina/script_utils/binder_inference_run.sh /path/to/finetuned_lora_merged.ckpt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAPROTEINA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${LAPROTEINA_ROOT}"

if [[ -n "${1:-}" ]]; then
  FT_CKPT_PATH="$1"
  FT_CKPT_DIR="$(dirname "${FT_CKPT_PATH}")"
  FT_CKPT_NAME="$(basename "${FT_CKPT_PATH}")"
  echo "=== Generating binder samples with checkpoint: ${FT_CKPT_PATH} ==="
  python proteinfoundation/generate.py \
    --config_name inference_motif_idx_aa_binder \
    ckpt_path="${FT_CKPT_DIR}" \
    ckpt_name="${FT_CKPT_NAME}"
else
  echo "=== Generating binder samples with LD4 (default checkpoint) ==="
  python proteinfoundation/generate.py \
    --config_name inference_motif_idx_aa_binder
fi

echo "Output: ./inference/inference_motif_idx_aa_binder_TARGET_BINDER_AA/"
