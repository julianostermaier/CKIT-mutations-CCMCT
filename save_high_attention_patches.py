from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_Features_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--feature_data_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--drop_out', action='store_true', default=False, 
                    help='whether model uses dropout')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))
args.n_classes=2

settings = {'split': args.split,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    for ckpt_idx in range(len(ckpt_paths)):
        print('fold {}'.format(ckpt_idx))
        split_csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
        dataset = Generic_Features_Dataset(data_csv_path ='dataset_csv/ckit_data.csv',
                        split_csv_path='splits/{}'.format(split_csv_path),
                        data_dir= args.feature_data_dir, 
                        label_dict = {'POSITIVE': 1, 'NEGATIVE': 0})
        high_attention_patches(dataset, args, ckpt_paths[ckpt_idx], args.data_dir, args.save_dir)
