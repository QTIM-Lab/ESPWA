#!/bin/bash
# Score slides with the trained fold models.
#
# Held-out test fold of each model (cross-validation predictions):
#   TASK=task_ER_positive_vs_negative LABELS_CSV=... FEATURES_DIR=... bash scripts/evaluate.sh
#
# Every slide of an external cohort with each of the ten models:
#   SPLIT=all SAVE_CODE=external TASK=... LABELS_CSV=external.csv FEATURES_DIR=... bash scripts/evaluate.sh
#
# Writes eval_results/EVAL_<SAVE_CODE>/fold_<k>.csv with slide_id, Y, Y_hat and,
# for the classification task, p_0 and p_1.
# Run from the repository root.
set -euo pipefail
: "${TASK:?set TASK}" "${LABELS_CSV:?set LABELS_CSV}" "${FEATURES_DIR:?set FEATURES_DIR}"
RESULTS_DIR=${RESULTS_DIR:-results}
SPLIT=${SPLIT:-test}
SAVE_CODE=${SAVE_CODE:-${TASK}_${SPLIT}}
FOLD=${FOLD:--1}   # -1 evaluates all ten folds

python eval.py \
    --task "$TASK" \
    --csv_path "$LABELS_CSV" \
    --data_root_dir "$FEATURES_DIR" \
    --results_dir "$RESULTS_DIR" \
    --models_exp_code "${TASK}_s1" \
    --save_exp_code "$SAVE_CODE" \
    --split "$SPLIT" \
    --k 10 --fold "$FOLD" \
    --embed_dim 768 \
    --drop_out 0.25
