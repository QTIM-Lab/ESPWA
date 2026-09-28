"""
ESPWA task definitions, shared by create_splits_seq.py, main.py and eval.py so
that splits, training and evaluation always read the same label table.

task_ER_positive_vs_negative
    binary ER status by IHC from the `label` column (0 = negative, 1 = positive);
    slides whose ER status is unknown must be left empty and are dropped
task_ERP_regression
    quantitative ER expression (fraction of ER-positive tumour cells, 0-1) from
    the `ERP_label` column; slides without a measured value (empty or 0) are
    dropped
"""
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset


def drop_missing_labels(label_col):
    def preprocess(df):
        missing = df[label_col].isna()
        if missing.any():
            print('dropping {} slides with no {}'.format(int(missing.sum()), label_col))
        return df[~missing].reset_index(drop=True)
    return preprocess

TASKS = {
    'task_ER_positive_vs_negative': {
        'n_classes': 2,
        'is_regression': False,
        'dataset_kwargs': {
            'label_dict': {0: 0, 1: 1},
            'label_col': 'label',
            'custom_preprocessing': drop_missing_labels('label'),
        },
    },
    'task_ERP_regression': {
        'n_classes': 1,
        'is_regression': True,
        'dataset_kwargs': {
            'label_dict': {},
            'label_col': 'ERP_label',
            'exclude_nulls': True,
            'is_regression': True,
        },
    },
}


def build_dataset(task, csv_path, data_dir=None, seed=1, print_info=True, patient_strat=True):
    """
    Build the dataset for an ESPWA task.

    csv_path: label table with at least slide_id, case_id and the task's label column
    data_dir: directory of per-slide feature .h5 files; None builds a label-only
        dataset, which is all create_splits_seq.py needs
    patient_strat: split by patient (needed to create splits); evaluation turns it
        off so that iterating the full dataset yields every slide
    Returns the dataset, the number of classes and whether the task is a regression.
    """
    if task not in TASKS:
        raise NotImplementedError('unknown task: {}'.format(task))
    spec = TASKS[task]
    kwargs = dict(csv_path=csv_path, shuffle=False, seed=seed, print_info=print_info,
                  patient_strat=patient_strat, ignore=[], **spec['dataset_kwargs'])
    if data_dir is None:
        dataset = Generic_WSI_Classification_Dataset(**kwargs)
    else:
        dataset = Generic_MIL_Dataset(data_dir=data_dir, **kwargs)
    return dataset, spec['n_classes'], spec['is_regression']
