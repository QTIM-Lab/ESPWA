import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_MIL(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.LongTensor(np.array([item[1] for item in batch]))
	return [img, label]

def collate_MIL_regression(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.FloatTensor(np.array([item[1] for item in batch]))
	return [img, label]

def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]


def get_simple_loader(dataset, batch_size=1, num_workers=1, is_regression=False):
	kwargs = {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
	collate_fn = collate_MIL_regression if is_regression else collate_MIL
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_fn, **kwargs)
	return loader 

def get_split_loader(split_dataset, training = False, testing = False, weighted = False, is_regression = False):
	"""
		return either the validation loader or training loader 
	"""
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}

	# Choose the appropriate collate function based on whether it's a regression task
	collate_fn = collate_MIL_regression if is_regression else collate_MIL

	if not testing:
		if training:
			if weighted:
				weights = make_weights_for_balanced_classes_split(split_dataset)
				loader = DataLoader(split_dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_fn, **kwargs)	
			else:
				loader = DataLoader(split_dataset, batch_size=1, sampler = RandomSampler(split_dataset), collate_fn = collate_fn, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate_fn, **kwargs)

	else:
		ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
		loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate_fn, **kwargs )

	return loader

def get_optim(model, args):
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	else:
		raise NotImplementedError
	return optimizer

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)

	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n

	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None, non_overlapping_splits = False):
	"""
	Generate splits for k-fold cross-validation.

	When non_overlapping_splits is True, this function ensures:
	1. Each sample appears in exactly one test set across all k folds
	2. All samples are used at least once in a test set

	Args:
		cls_ids: List of arrays containing indices for each class
		val_num: Number of validation samples per class
		test_num: Number of test samples per class
		samples: Total number of samples
		n_splits: Number of splits (k folds)
		seed: Random seed
		label_frac: Fraction of labels to use
		custom_test_ids: Custom test IDs (if provided)
		non_overlapping_splits: Whether to ensure non-overlapping test sets
	"""
	indices = np.arange(samples).astype(int)

	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	np.random.seed(seed)

	# Pre-assign each sample to exactly one test fold if using non-overlapping splits
	all_test_folds = None
	if non_overlapping_splits and custom_test_ids is None:
		# For each class, distribute samples evenly across test folds
		all_test_folds = [[] for _ in range(n_splits)]

		for c in range(len(cls_ids)):
			class_indices = cls_ids[c].copy()
			np.random.shuffle(class_indices)

			# Calculate samples per fold for this class
			# Ensure we use all samples by distributing them evenly
			samples_per_fold = len(class_indices) // n_splits
			remainder = len(class_indices) % n_splits

			# Distribute samples to folds
			start_idx = 0
			for i in range(n_splits):
				# Add extra sample from remainder if available
				extra = 1 if i < remainder else 0
				fold_size = samples_per_fold + extra

				# Get indices for this fold
				if fold_size > 0:
					fold_indices = class_indices[start_idx:start_idx + fold_size]
					all_test_folds[i].extend(fold_indices)
					start_idx += fold_size

	# Generate each split
	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []

		if custom_test_ids is not None:
			# Use custom test IDs if provided
			all_test_ids.extend(custom_test_ids)
		elif non_overlapping_splits and all_test_folds is not None:
			# Use pre-assigned test fold for this split when using non-overlapping splits
			all_test_ids = all_test_folds[i]
		else:
			# random test samples for each fold (upstream CLAM behaviour)
			for c in range(len(test_num)):
				possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class

				# For validation
				val_ids = np.random.choice(possible_indices, val_num[c], replace=False)
				remaining_ids = np.setdiff1d(possible_indices, val_ids)
				all_val_ids.extend(val_ids)

				# For test
				test_ids = np.random.choice(remaining_ids, test_num[c], replace=False)
				all_test_ids.extend(test_ids)

				# Skip the rest of the loop since we've already selected validation samples
				continue

		# For non-overlapping splits or custom test IDs, we need to select validation samples
		if non_overlapping_splits or custom_test_ids is not None:
			# Remove test IDs from available indices for this fold
			available_indices = np.setdiff1d(indices, all_test_ids)

			# Select validation samples from remaining indices
			for c in range(len(val_num)):
				possible_indices = np.intersect1d(cls_ids[c], available_indices)

				# Ensure we don't try to select more validation samples than available
				val_count = min(val_num[c], len(possible_indices))

				if val_count > 0:
					val_ids = np.random.choice(possible_indices, val_count, replace=False)
					all_val_ids.extend(val_ids)
					available_indices = np.setdiff1d(available_indices, val_ids)

			# Remaining samples go to training
			if label_frac == 1:
				sampled_train_ids = available_indices.tolist()
			else:
				# Sample a fraction of the remaining indices for training
				for c in range(len(cls_ids)):
					possible_indices = np.intersect1d(cls_ids[c], available_indices)
					sample_num = math.ceil(len(possible_indices) * label_frac)
					if sample_num > 0:
						sampled_indices = np.random.choice(possible_indices, sample_num, replace=False)
						sampled_train_ids.extend(sampled_indices)
		else:
			# training samples for the random-split case
			for c in range(len(cls_ids)):
				possible_indices = np.intersect1d(cls_ids[c], indices)
				# Remove validation and test samples
				val_test_ids = np.concatenate([
					np.intersect1d(all_val_ids, possible_indices),
					np.intersect1d(all_test_ids, possible_indices)
				])
				remaining_ids = np.setdiff1d(possible_indices, val_test_ids)

				if label_frac == 1:
					sampled_train_ids.extend(remaining_ids)
				else:
					sample_num = math.ceil(len(remaining_ids) * label_frac)
					slice_ids = np.arange(sample_num)
					sampled_train_ids.extend(remaining_ids[slice_ids])

		yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

def calculate_pcc(Y_hat, Y):
	"""Pearson correlation between predictions and labels (0.0 if undefined)"""
	Y_hat = np.asarray(Y_hat, dtype=float).flatten()
	Y = np.asarray(Y, dtype=float).flatten()
	if len(Y) < 2 or Y_hat.std() == 0 or Y.std() == 0:
		return 0.0
	return float(np.corrcoef(Y_hat, Y)[0, 1])

def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))                                           
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx)                        
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()

		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)
