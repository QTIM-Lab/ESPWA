#!/bin/bash
#SBATCH --job-name=espwa
#SBATCH --array=0-9
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# Train ESPWA on the 10 cross-validation folds.
#
#   TASK=task_ER_positive_vs_negative LABELS_CSV=... FEATURES_DIR=... bash scripts/train.sh
#   TASK=task_ERP_regression          LABELS_CSV=... FEATURES_DIR=... sbatch scripts/train.sh
#
# Under SLURM each array task trains one fold; run with bash, it trains all ten in turn.
# Run from the repository root.
set -euo pipefail
: "${TASK:?set TASK}" "${LABELS_CSV:?set LABELS_CSV}" "${FEATURES_DIR:?set FEATURES_DIR}"
SPLITS_DIR=${SPLITS_DIR:-splits}
RESULTS_DIR=${RESULTS_DIR:-results}

FOLD_ARGS=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    FOLD_ARGS=(--fold "$SLURM_ARRAY_TASK_ID")
fi

# CLAM-SB (small), CONCH v1.5 features, Adam 1e-4, weight decay 1e-5, dropout 0.25,
# 0.7 bag loss + 0.3 instance loss with B=8, up to 200 epochs with early stopping.
# The regression task switches to the CLAM regression head and MSE on its own.
python main.py \
    --task "$TASK" \
    --csv_path "$LABELS_CSV" \
    --data_root_dir "$FEATURES_DIR" \
    --split_dir "$SPLITS_DIR/$TASK" \
    --results_dir "$RESULTS_DIR" \
    --exp_code "$TASK" \
    --k 10 "${FOLD_ARGS[@]}" \
    --embed_dim 768 \
    --model_type clam_sb --model_size small \
    --bag_loss ce --inst_loss ce --bag_weight 0.7 --B 8 \
    --drop_out 0.25 --lr 1e-4 --reg 1e-5 --opt adam \
    --max_epochs 200 --early_stopping --patience 20 --stop_epoch 50 \
    --seed 1 --log_data
