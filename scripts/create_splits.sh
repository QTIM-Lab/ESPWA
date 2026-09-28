#!/bin/bash
# 10-fold patient-level splits for both ESPWA tasks: every patient is in exactly one
# test fold, with 10% of patients for validation and 10% for test in each fold.
#
#   LABELS_CSV=/path/to/labels.csv SPLITS_DIR=splits bash scripts/create_splits.sh
# Run from the repository root.
set -euo pipefail
: "${LABELS_CSV:?set LABELS_CSV}"
SPLITS_DIR=${SPLITS_DIR:-splits}

for TASK in task_ER_positive_vs_negative task_ERP_regression; do
    python create_splits_seq.py \
        --task "$TASK" \
        --csv_path "$LABELS_CSV" \
        --split_dir "$SPLITS_DIR/$TASK" \
        --k 10 --val_frac 0.1 --test_frac 0.1 --seed 1
done
