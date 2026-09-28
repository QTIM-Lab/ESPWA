# ESPWA

**E**strogen Receptor **S**tatus **P**rediction for Haitian patients using deep learning-enabled
H&E **W**hole Slide Imaging **A**nalysis ("hope" in Haitian Creole).

Code for *ESPWA: a deep learning tool to inform precision-based use of endocrine therapy in
resource-limited settings* ([bioRxiv 10.1101/2025.08.27.672012](https://doi.org/10.1101/2025.08.27.672012)).

ESPWA predicts estrogen receptor (ER) status from H&E whole-slide images (WSIs) of breast
cancer patients treated at Zanmi Lasante, Haiti. It has two heads trained on the same slides:

| task | target | model |
|---|---|---|
| `task_ER_positive_vs_negative` | ER status by IHC (negative / positive) | CLAM single-branch, cross-entropy |
| `task_ERP_regression` | fraction of ER-positive tumour cells (0–1) | CLAM attention + linear regression head, MSE |

This repository holds the pipeline only: patch features, patient-level splits, training,
inference and attention heatmaps. The patient data are not public (see [Data](#data)).

## Pipeline

```
WSI ─▶ TRIDENT: tissue segmentation, 20x / 256 px tiles ─▶ CONCH v1.5 patch features (768-d)
                                                                    │
labels.csv ─▶ create_splits_seq.py (10 patient-level folds) ───────┤
                                                                    ▼
                                  main.py: CLAM, one model per fold ─▶ eval.py: per-slide predictions
                                                                    └─▶ create_heatmaps.py: attention maps
```

## Installation

```bash
pip install -r requirements.txt
```

Feature extraction uses [TRIDENT](https://github.com/mahmoodlab/TRIDENT) in its own
environment. The CONCH v1.5 weights are gated on Hugging Face: request access to
[`MahmoodLab/conchv1_5`](https://huggingface.co/MahmoodLab/conchv1_5) for feature extraction
and to [`MahmoodLab/TITAN`](https://huggingface.co/MahmoodLab/TITAN), through which
`create_heatmaps.py` loads the same encoder, then `huggingface-cli login`.

All commands below run from the repository root.

## Data

The Zanmi Lasante cohort cannot be shared publicly. Requests for raw and analyzed data are
reviewed by the Massachusetts General Hospital Institutional Review Board; data that can be
shared require its approval and a Material Transfer Agreement, and are de-identified.

To run ESPWA on your own slides, provide a label table (`labels.csv`) with one row per slide:

| column | content |
|---|---|
| `slide_id` | slide file name without extension; must match the feature file `<slide_id>.h5` |
| `case_id` | patient identifier; all slides of a patient share it and stay in the same split |
| `label` | ER status by IHC: `1` positive, `0` negative, **empty if unknown** |
| `ERP_label` | fraction of ER-positive tumour cells (0–1), **empty if not measured** |

Slides with an empty `label` are left out of the classification task, and slides with an empty
or zero `ERP_label` are left out of the regression task.

## 1. Patch features

```bash
WSI_DIR=/data/slides JOB_DIR=/data/features TRIDENT_DIR=/opt/TRIDENT \
    bash scripts/extract_features.sh
```

TRIDENT segments tissue with its HEST segmenter, tiles it into non-overlapping 256 px patches
at 20x, and encodes each patch with CONCH v1.5. Features are written to
`$JOB_DIR/20x_256px_0px_overlap/features_conch_v15/<slide_id>.h5`, with datasets `features`
(N × 768) and `coords` (N × 2, level-0 pixels). That directory is `FEATURES_DIR` below.

## 2. Splits

```bash
LABELS_CSV=/data/labels.csv bash scripts/create_splits.sh
```

This writes `splits/<task>/splits_{0..9}.csv`. Patients are split, not slides. Every patient
appears in exactly one of the ten test folds. In each fold, 10% of the patients form the test
set, 10% are drawn from the rest for validation, and the remaining ~80% are used for training.
Both tasks are split from the same `labels.csv`, and each split is stratified by class for
classification. The splits do not change unless `--seed` changes (default 1).

## 3. Training

```bash
TASK=task_ER_positive_vs_negative LABELS_CSV=/data/labels.csv FEATURES_DIR=/data/features/... \
    sbatch scripts/train.sh     # SLURM array: one fold per task
# or: bash scripts/train.sh     # all ten folds in sequence
```

Replace `TASK` with `task_ERP_regression` to train the regression head. Hyperparameters, as set
in `scripts/train.sh`:

- CLAM-SB (small): 768 → 512 fc, gated attention with 256 hidden units, dropout 0.25
- Adam, learning rate 1e-4, weight decay 1e-5, batch size 1 (one slide)
- classification: 0.7 × bag cross-entropy + 0.3 × instance cross-entropy with B = 8
- regression: MSE on the attention-pooled slide embedding, with no instance-level loss
- up to 200 epochs, early stopping on validation loss with patience 20, never before epoch 50
- seed 1

Each fold writes the following to `results/<task>_s1/`:

- `s_<k>_checkpoint.pt`, the weights with the lowest validation loss
- `splits_<k>.csv`
- `split_<k>_results.pkl`, the per-slide predictions on that fold's test set
- a TensorBoard log

## 4. Inference

Cross-validation predictions, where each model scores its own held-out test fold:

```bash
TASK=task_ER_positive_vs_negative LABELS_CSV=/data/labels.csv FEATURES_DIR=/data/features/... \
    bash scripts/evaluate.sh
```

External cohort, where each of the ten fold models scores every slide in the table:

```bash
SPLIT=all SAVE_CODE=external TASK=task_ER_positive_vs_negative \
    LABELS_CSV=/data/external_labels.csv FEATURES_DIR=/data/external_features/... \
    bash scripts/evaluate.sh
```

Predictions are written to `eval_results/EVAL_<SAVE_CODE>/fold_<k>.csv`:

- classification: `slide_id`, `Y` (label), `Y_hat` (predicted class), `p_0`, `p_1`
- regression: `slide_id`, `Y`, `Y_hat` (predicted ER fraction)

Set `FOLD=<k>` to evaluate a single fold.

## 5. Attention heatmaps

Fill in the `<...>` fields of `heatmaps/configs/espwa_er.yaml`: slide directory, a process list
placed in `heatmaps/process_lists/` with a `slide_id` column, and a checkpoint. Then run:

```bash
python create_heatmaps.py --config_file espwa_er.yaml
```

The config re-extracts patches at the magnification used for training, 256 px at 20x. On 40x
scans it reads 512 px tiles at level 0 and downsamples them by 2; on 20x scans, set
`patch_size: 256` and `custom_downsample: 1`.

## Relation to CLAM

This code is built on [CLAM](https://github.com/mahmoodlab/CLAM) (Lu et al., *Nat. Biomed. Eng.*
2021) at commit `53e2409`. The first commit after the initial one imports that code unchanged,
so `git diff` against it shows every ESPWA change:

- `models/model_clam.py`:
  - new `CLAM_REG` regression head
  - instance sampling uses `min(B, bag size)` patches, so bags smaller than B are supported
- `utils/core_utils.py`, `utils/eval_utils.py`: regression training, validation and inference; configurable early stopping
- `utils/utils.py`: patient-level splits in which each patient is tested exactly once; regression data loaders
- `dataset_modules/dataset_generic.py`:
  - reads TRIDENT `.h5` features
  - drops slides with missing labels
  - regression datasets
- `dataset_modules/espwa_tasks.py`: the two ESPWA tasks, shared by the split, training and evaluation scripts
- `main.py`, `eval.py`, `create_splits_seq.py`: label table and feature directory passed on the command line; single-fold runs
- `create_heatmaps.py`, `vis_utils/heatmap_utils.py`: runs without an interactive prompt; percentile scoring works on arrays
- NumPy 2 compatibility

The upstream patching and feature-extraction scripts are not included, because features come from TRIDENT.

## License

GPL-3.0, inherited from CLAM. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{pulidoarias2025espwa,
  title   = {{ESPWA}: a deep learning tool to inform precision-based use of endocrine therapy in resource-limited settings},
  author  = {Pulido-Arias, Dagoberto and Henderson, Rebecca and Patel, Milit and Millien, Christophe and Lomil, Joarly and Jose, Marie Djenane and Flambert, Gabriel and Bontemps, Jean and Georges, Emmanuel and Gunturi, Alekhya and Shah, Palak and Goncalves, Tiago and Kalpathy-Cramer, Jayashree and Gerstner, Elizabeth and Wander, Seth A. and Sirintrapun, S. Joe and Sgroi, Dennis and Jeronimo, Jose and Castle, Philip E. and Landgraf, Kenneth and Brown, Ali and Fadelu, Temidayo and Shulman, Lawrence N. and Guttag, John and Milner, Dan and Brock, Jane and Bridge, Christopher P. and Kim, Albert E.},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.08.27.672012}
}
```

Please also cite CLAM, CONCH and TRIDENT if you use this code.
