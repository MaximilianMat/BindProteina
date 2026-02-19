#!/usr/bin/env bash
set -euo pipefail

# Staged single-GPU fine-tuning schedule for direct binder generation.
# Usage:
#   bash script_utils/binder_ft_stage_schedule.sh smoke
#   bash script_utils/binder_ft_stage_schedule.sh main
#   bash script_utils/binder_ft_stage_schedule.sh unfreeze

PHASE="${1:-smoke}"

case "${PHASE}" in
  smoke)
    python proteinfoundation/train.py \
      --config-name training_local_latents_binder_ft \
      +single=true \
      run_name_=binder_ft_relpos_lora_smoke \
      opt.max_epochs=1 \
      opt.val_check_interval=100 \
      log.checkpoint=false \
      log.log_wandb=false \
      dataset.datamodule.dataselector.fraction=0.0001
    ;;

  main)
    # 50/50 curriculum via alternating resumed runs:
    # 1) multimer (binder objective) -> 2) motif single-chain (stability retention)
    # Re-run this phase repeatedly until desired wall-clock budget is reached.
    python proteinfoundation/train.py \
      --config-name training_local_latents_binder_ft \
      +single=true \
      run_name_=binder_ft_relpos_lora_2 \
      dataset.datamodule.dataselector.fraction=0.1

    python proteinfoundation/train.py \
      --config-name training_local_latents_binder_ft \
      +single=true \
      run_name_=binder_ft_relpos_lora_2 \
      dataset=pdb/pdb_train_motif_aa \
      dataset.datamodule.dataselector.fraction=0.1
    ;;

  unfreeze)
    python proteinfoundation/train.py \
      --config-name training_local_latents_binder_ft \
      +single=true \
      run_name_=binder_ft_relpos_partial_unfreeze \
      lora.r=null \
      opt.lr=0.00002 \
      opt.accumulate_grad_batches=2 \
      dataset.datamodule.dataselector.fraction=0.1
    ;;

  *)
    echo "Unknown phase: ${PHASE}"
    echo "Expected one of: smoke, main, unfreeze"
    exit 1
    ;;
esac
