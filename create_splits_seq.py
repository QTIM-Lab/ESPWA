import os
import pandas as pd
from dataset_modules.dataset_generic import save_splits
from dataset_modules.espwa_tasks import TASKS, build_dataset
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating patient-level splits for ESPWA')
parser.add_argument('--csv_path', type=str, required=True,
                    help='label table (slide_id, case_id, label, ERP_label)')
parser.add_argument('--split_dir', type=str, default=None,
                    help='output directory (default: splits/<task>_<label_frac*100>)')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=list(TASKS), required=True)
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')

args = parser.parse_args()

dataset, args.n_classes, _ = build_dataset(args.task, args.csv_path, seed=args.seed)

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        if args.split_dir is not None and len(label_fracs) == 1:
            split_dir = args.split_dir
        else:
            split_dir = os.path.join(args.split_dir or 'splits', str(args.task) + '_{}'.format(int(lf * 100)))
        os.makedirs(split_dir, exist_ok=True)
        # every patient lands in exactly one test fold; validation is drawn from the rest
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf,
                              non_overlapping_splits=True)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



