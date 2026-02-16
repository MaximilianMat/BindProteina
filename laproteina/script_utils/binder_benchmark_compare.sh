#!/usr/bin/env bash
set -euo pipefail

# Compare LD4 baseline vs fine-tuned checkpoint on the same binder task.
#
# Usage:
#   bash laproteina/script_utils/binder_benchmark_compare.sh /path/to/finetuned.ckpt
#
# Optional scoring:
#   SCORE=1 SETTINGS=./settings_target/my_target.json FILTERS=./settings_filters/laProteina_filters.json \
#   bash laproteina/script_utils/binder_benchmark_compare.sh /path/to/finetuned.ckpt

FT_CKPT_PATH="${1:-}"
if [[ -z "${FT_CKPT_PATH}" ]]; then
  echo "Usage: bash laproteina/script_utils/binder_benchmark_compare.sh /path/to/finetuned.ckpt"
  exit 1
fi

if [[ ! -f "${FT_CKPT_PATH}" ]]; then
  echo "Fine-tuned checkpoint not found: ${FT_CKPT_PATH}"
  exit 1
fi

FT_CKPT_DIR="$(dirname "${FT_CKPT_PATH}")"
FT_CKPT_NAME="$(basename "${FT_CKPT_PATH}")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAPROTEINA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${LAPROTEINA_ROOT}"

TMP_OUT="./inference/inference_motif_idx_aa_binder_TARGET_BINDER_AA"
BASELINE_OUT="./inference/benchmark_ld4_pretrained"
FINETUNED_OUT="./inference/benchmark_ld4_finetuned"

rm -rf "${TMP_OUT}" "${BASELINE_OUT}" "${FINETUNED_OUT}"

echo "=== Generating baseline samples with LD4 ==="
python proteinfoundation/generate.py \
  --config_name inference_motif_idx_aa_binder
mv "${TMP_OUT}" "${BASELINE_OUT}"

echo "=== Generating fine-tuned samples ==="
python proteinfoundation/generate.py \
  --config_name inference_motif_idx_aa_binder \
  ckpt_path="${FT_CKPT_DIR}" \
  ckpt_name="${FT_CKPT_NAME}"
mv "${TMP_OUT}" "${FINETUNED_OUT}"

echo "Baseline samples:  ${BASELINE_OUT}"
echo "Fine-tuned samples: ${FINETUNED_OUT}"

if [[ "${SCORE:-0}" == "1" ]]; then
  SETTINGS_PATH="${SETTINGS:-}"
  FILTERS_PATH="${FILTERS:-./settings_filters/laProteina_filters.json}"
  if [[ -z "${SETTINGS_PATH}" ]]; then
    echo "SCORE=1 requires SETTINGS=/path/to/settings.json"
    exit 1
  fi

  echo "=== Scoring baseline samples ==="
  python laproteina_scoring/score.py \
    --input_dir "${BASELINE_OUT}" \
    --output_dir "./benchmark_scores/pretrained_ld4" \
    --settings "${SETTINGS_PATH}" \
    --filters "${FILTERS_PATH}"

  echo "=== Scoring fine-tuned samples ==="
  python laproteina_scoring/score.py \
    --input_dir "${FINETUNED_OUT}" \
    --output_dir "./benchmark_scores/finetuned" \
    --settings "${SETTINGS_PATH}" \
    --filters "${FILTERS_PATH}"

  echo "Scoring done. Compare CSVs in:"
  echo "  ./benchmark_scores/pretrained_ld4"
  echo "  ./benchmark_scores/finetuned"
fi
