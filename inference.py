from __future__ import print_function

import numpy as np

import argparse

import torch
import torch.nn as nn
import pdb
import os
import glob
import pandas as pd
from utils.utils import *
from math import floor
from utils.eval_utils import initiate_model as initiate_model
from models.model_clam import CLAM_MB, CLAM_SB
from models.resnet_custom import resnet50_baseline
from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
from wsi_core.batch_process_utils import initialize_df
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
from wsi_core.wsi_utils import sample_rois
from utils.file_utils import save_hdf5

def infer_single_slide(model, features, label, reverse_label_dict, k=1):
	features = features.to(device)
	with torch.no_grad():
		if isinstance(model, (CLAM_SB, CLAM_MB)):
			model_results_dict = model(features)
			logits, Y_prob, Y_hat, A, _ = model(features)
			Y_hat = Y_hat.item()

			if isinstance(model, (CLAM_MB,)):
				A = A[Y_hat]

			A = A.view(-1, 1).cpu().numpy()

		else:
			raise NotImplementedError

		print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))	
		
		probs, ids = torch.topk(Y_prob, k)
		probs = probs[-1].cpu().numpy()
		ids = ids[-1].cpu().numpy()
		preds_str = np.array([reverse_label_dict[idx] for idx in ids])

	return ids, preds_str, probs, A

def load_params(df_entry, params):
	for key in params.keys():
		if key in df_entry.index:
			dtype = type(params[key])
			val = df_entry[key] 
			val = dtype(val)
			if isinstance(val, str):
				if len(val) > 0:
					params[key] = val
			elif not np.isnan(val):
				params[key] = val
			else:
				pdb.set_trace()

	return params


def inference(slide, data_dir, res_dir='inference'):
	'''
	perform inference on a single slide, returns heatmap (downsampling factor 15) and the predicted label
	args:
		slide (string): slide file name 
		data_dir (dict): directory where slide is located
		res_dir: directory where intermediate results are saved; default '/inference'
	'''
	config_dict = {
    'exp_arguments': {
		'n_classes': 2,
		'batch_size': 128
		}, 
	'data_arguments': {
		'label_dict': {'NEGATIVE': 0, 'POSITIVE': 1}
		}, 
	'patching_arguments': {
		'patch_size': 256, 
		'overlap': 0, 
		'patch_level': 0, 
		'custom_downsample': 1
		}, 
	'model_arguments': {
		'resnet_weights':'pretraining/finetuning/finetuning_high_att/checkpoint_epoch_98.pt',
		'clam_weights': 'results/c-kit-mutation-moco-imgnet-lrlower-finetuned_s1/s_4_checkpoint.pt', 
		'model_type': 'clam_sb', 
		'initiate_fn': 'initiate_model', 
		'model_size': 'big', 
		'drop_out': True
		}, 
	'heatmap_arguments': {
		'vis_level': 1, 
		'alpha': 1, 
		'blank_canvas': True, 
		'save_orig': False, 
		'save_ext': 'jpg', 
		'use_ref_scores': True, 
		'blur': False, 
		'use_center_shift': True, 
		'use_roi': False, 
		'calc_heatmap': True, 
		'binarize': False, 
		'binary_thresh': -1, 
		'custom_downsample': 15, 
		'cmap': 'jet'
		}
	}


	args = config_dict
	patch_args = argparse.Namespace(**args['patching_arguments'])
	data_args = argparse.Namespace(**args['data_arguments'])
	model_args = args['model_arguments']
	model_args.update({'n_classes': args['exp_arguments']['n_classes']})
	model_args = argparse.Namespace(**model_args)
	exp_args = argparse.Namespace(**args['exp_arguments'])
	heatmap_args = argparse.Namespace(**args['heatmap_arguments'])

	patch_size = tuple([patch_args.patch_size for i in range(2)])
	step_size = tuple((np.array(patch_size) * (1-patch_args.overlap)).astype(int))
	print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))

	
	def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False, 
					  'keep_ids': 'none', 'exclude_ids':'none'}
	def_filter_params = {'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10}
	def_vis_params = {'vis_level': -1, 'line_thickness': 250}
	def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	print('\ninitializing model from checkpoint')
	ckpt_path = model_args.clam_weights
	print('\nckpt path: {}'.format(ckpt_path))
	
	if model_args.initiate_fn == 'initiate_model':
		model =  initiate_model(model_args, ckpt_path)
	else:
		raise NotImplementedError

	model_path = model_args.resnet_weights
	feature_extractor = resnet50_baseline(pretrained=True, model_path=model_path)
	feature_extractor.eval()
	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print('Done!')

	label_dict =  data_args.label_dict
	class_labels = list(label_dict.keys())
	class_encodings = list(label_dict.values())
	reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 

	if torch.cuda.device_count() > 1:
		device_ids = list(range(torch.cuda.device_count()))
		feature_extractor = nn.DataParallel(feature_extractor, device_ids=device_ids).to('cuda:0')
	else:
		feature_extractor = feature_extractor.to(device)

	blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size, 
	'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

	top_left = None
	bot_right = None

	if isinstance(data_dir, str):
		# find files recursively in subfolders
		slide_path = glob.glob(f'{data_dir}/**/{slide}*', 
				recursive = True)[0]

	slide_id = slide.replace('.svs', '')
	r_slide_save_dir = os.path.join(res_dir, slide_id)
	os.makedirs(r_slide_save_dir, exist_ok=True)
	mask_file = os.path.join(r_slide_save_dir, slide_id+'_mask.pkl')
	
	# Load segmentation and filter parameters
	seg_params = def_seg_params.copy()
	filter_params = def_filter_params.copy()
	vis_params = def_vis_params.copy()

	keep_ids = str(seg_params['keep_ids'])
	if len(keep_ids) > 0 and keep_ids != 'none':
		seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
	else:
		seg_params['keep_ids'] = []

	exclude_ids = str(seg_params['exclude_ids'])
	if len(exclude_ids) > 0 and exclude_ids != 'none':
		seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
	else:
		seg_params['exclude_ids'] = []

	for key, val in seg_params.items():
		print('{}: {}'.format(key, val))

	for key, val in filter_params.items():
		print('{}: {}'.format(key, val))

	for key, val in vis_params.items():
		print('{}: {}'.format(key, val))
	
	print('Initializing WSI object')
	wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
	print('Done!')

	wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]

	# the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
	vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

	block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
	
	features_path = os.path.join(r_slide_save_dir, slide_id+'.pt')
	h5_path = os.path.join(r_slide_save_dir, slide_id+'.h5')

	t = transforms.Compose(
		[transforms.ToTensor(),
		transforms.Normalize(mean = [0.7785, 0.6139, 0.7132], 
							std = [0.1942, 0.2412, 0.1882])])
	
	##### check if h5_features_file exists ######
	if not os.path.isfile(h5_path) :
		print(t)
		_, _, wsi_object = compute_from_patches(wsi_object=wsi_object, 
										model=model, 
										feature_extractor=feature_extractor, 
										batch_size=exp_args.batch_size, **blocky_wsi_kwargs, 
										attn_save_path=None, feat_save_path=h5_path, 
										ref_scores=None, t=t)
		
	##### check if pt_features_file exists ######
	if not os.path.isfile(features_path):
		file = h5py.File(h5_path, "r")
		features = torch.tensor(file['features'][:])
		torch.save(features, features_path)
		file.close()

	# load features 
	features = torch.load(features_path)
	
	wsi_object.saveSegmentation(mask_file)
	Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, features, '', reverse_label_dict, exp_args.n_classes)
	del features
	
	file = h5py.File(h5_path, "r")
	dset = A
	coord_dset = file['coords'][:]
	scores = dset[:]
	coords = coord_dset[:]

	heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': heatmap_args.vis_level, 'blur': heatmap_args.blur, 'custom_downsample': heatmap_args.custom_downsample}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
	heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object,  
							cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, **heatmap_vis_args, 
							binarize=heatmap_args.binarize, 
							blank_canvas=heatmap_args.blank_canvas,
							thresh=heatmap_args.binary_thresh,  patch_size = vis_patch_size,
							overlap=patch_args.overlap, 
							top_left=top_left, bot_right = bot_right)
	
	# delete intermediate results from results folder
	files = glob.glob(r_slide_save_dir + '/*')
	for f in files:
		os.remove(f)
	os.rmdir(r_slide_save_dir)

	return heatmap, Y_hats[0]


