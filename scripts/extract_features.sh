#!/bin/bash
# Tissue segmentation, 20x / 256 px tiling and CONCH v1.5 patch features with TRIDENT
# (https://github.com/mahmoodlab/TRIDENT, used at commit fe5075c).
#
#   WSI_DIR=/path/to/slides JOB_DIR=/path/to/features TRIDENT_DIR=/path/to/TRIDENT \
#     bash scripts/extract_features.sh
#
# Features land in $JOB_DIR/20x_256px_0px_overlap/features_conch_v15/<slide_id>.h5,
# which is the --data_root_dir for main.py and eval.py.
set -euo pipefail
: "${WSI_DIR:?set WSI_DIR}" "${JOB_DIR:?set JOB_DIR}" "${TRIDENT_DIR:?set TRIDENT_DIR}"

cd "$TRIDENT_DIR"
python run_batch_of_slides.py \
    --task all \
    --wsi_dir "$WSI_DIR" \
    --job_dir "$JOB_DIR" \
    --segmenter hest \
    --seg_conf_thresh 0.5 \
    --mag 20 \
    --patch_size 256 \
    --overlap 0 \
    --min_tissue_proportion 0 \
    --patch_encoder conch_v15 \
    --batch_size 32
