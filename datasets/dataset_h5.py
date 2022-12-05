from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
import helpers
import openslide
import glob
import random

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F

from PIL import Image
import h5py

from random import randrange
from pathlib import Path

def eval_transforms(pretrained='ImageNet'):
    if pretrained == 'ImageNet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif pretrained == 'MoCo':
        mean = (0.7785, 0.6139, 0.7132)
        std = (0.1942, 0.2412, 0.1882)
    else:
        mean = (0.5,0.5,0.5)
        std = (0.5,0.5,0.5)

    trnsfrms_val = transforms.Compose(
                    [
                     transforms.ToTensor(),
                     transforms.Normalize(mean = mean, std = std)
                    ]
                )

    return trnsfrms_val

class Whole_Slide_Bag(Dataset):
    def __init__(self,
        file_path,
        pretrained='ImageNet',
        custom_transforms=None,
        target_patch_size=-1,
        ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.pretrained=pretrained
        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size, target_patch_size)
        else:
            self.target_patch_size = None

        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['imgs']
            self.length = len(dset)

        self.summary()
            
    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['imgs']
        for name, value in dset.attrs.items():
            print(name, value)

        print('pretrained:', self.pretrained)
        print('transformations:', self.roi_transforms)
        if self.target_patch_size is not None:
            print('target_size: ', self.target_patch_size)

    def __getitem__(self, idx):
        with h5py.File(self.file_path,'r') as hdf5_file:
            img = hdf5_file['imgs'][idx]
            coord = hdf5_file['coords'][idx]
        
        img = Image.fromarray(img)
        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord

class Whole_Slide_Bag_FP(Dataset):
    def __init__(self,
        file_path,
        wsi,
        pretrained='ImageNet',
        custom_transforms=None,
        custom_downsample=1,
        target_patch_size=-1
        ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms or MoCo transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
            target_patch_size (int): Custom defined image size before embedding
        """
        self.pretrained=pretrained
        self.wsi = wsi
        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(dset)
            if target_patch_size > 0:
                self.target_patch_size = (target_patch_size, ) * 2
            elif custom_downsample > 1:
                self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
            else:
                self.target_patch_size = None
        self.summary()
            
    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['coords']
        for name, value in dset.attrs.items():
            print(name, value)

        print('\nfeature extraction settings')
        print('target patch size: ', self.target_patch_size)
        print('pretrained: ', self.pretrained)
        print('transformations: ', self.roi_transforms)

    def __getitem__(self, idx):
        with h5py.File(self.file_path,'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord

class Dataset_All_Bags(Dataset):

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df['slide_id'][idx]

class Dataset_All_Bags_label(Dataset):

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df['slide_id'][idx], self.df['label'][idx]


class HDF5Dataset(Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        wsi_path: Path to WSI's.
        recursive: If True, searches for h5 files in subdirectories.
        csv_path: if defined, only slides specified in the csv_file will be used in dataset (useful for train/test split).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, file_path, wsi_path, recursive, csv_path=None, transform=None, mag_prior=False):
        super().__init__()

        self.data_info = []
        self.mag_prior = mag_prior
        self.transform = transform
        print(self.mag_prior)

        # Search for all h5 and svs files
        w = Path(wsi_path)
        assert(w.is_dir())

        if csv_path:
            files_csv = pd.read_csv(csv_path)['slides']
            files = []
            for slide in files_csv:
                print(glob.glob('{}/**/patches/{}.h5'.format(file_path, slide), recursive=True))
                file = glob.glob('{}/**/patches/{}.h5'.format(file_path, slide), recursive=True)
                files.append(file[0])
        else:
            files = sorted(p.glob('**/*.h5'))

        wsis = sorted(w.glob('**/*.svs'))

        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')
            
        self.wsi_files = [str(f.resolve()) for f in wsis]

        for i, h5dataset_fp in enumerate(files):
            self._add_data_infos(str(h5dataset_fp), i)
            
    def __getitem__(self, index):
        
        if self.mag_prior:
            x_1, x_2 = self.get_data_mag(index)
            x = self.transform(x_1, x_2)
        else:
            x = self.get_data(index)
            x = self.transform(x)
            
        return x

    def __len__(self):
        return len(self.data_info)
    
    def _add_data_infos(self, file_path, i):
        
        with h5py.File(file_path, "r") as h5_file:
            # get metadata for WSI
            patch_level = h5_file['coords'].attrs['patch_level']
            patch_size = h5_file['coords'].attrs['patch_size']

            # iterate over coordinates of all patches in WSI
            print('WSI number of patches : {}'.format(h5_file['coords'].len()))
            for patch_idx in range(h5_file['coords'].len()):
                coord = h5_file['coords'][patch_idx]
                
                # add information to data_info
                wsi_path = self._find_wsi(file_path)
                if wsi_path != None:
                    self.data_info.append({'file_path': file_path, 'wsi_path': wsi_path, 'coords': coord, 'patch_level': patch_level, 'patch_size': patch_size})
                
    def _find_wsi(self, file_path):
        """match .h5 file to the corresponding wsi file_path
        """
        
        # returns only filename of file in file_path without .h5 ending
        file_name = file_path.split('/')[-1].split('.')[0]
        if len(file_name) < 1:
            raise Exception("could not extract file_name")
        
        # find wsi path for file_name
        wsi_match = [f for f in self.wsi_files if file_name in f]
            
        return wsi_match[0] if len(wsi_match)>0 else None

    def get_data(self, i, patch_coords=None, patch_level=None):
        """Call this function anytime you want to access a chunk of data
        """
        patch = self.data_info[i]
        
        wsi = openslide.open_slide(patch['wsi_path'])
        img = wsi.read_region(patch['coords'], patch['patch_level'], (patch['patch_size'], patch['patch_size'])).convert('RGB')

        return img

    def get_data_mag(self, i):
        """Call this function anytime you want to access a chunk of data
        """
        
        level_downsamples = {
            '0': 1,
            '1': 4,
            '2': 16,
            '3': 32
        }

        mag_1 = randrange(3)
        mag_2 = mag_1 + 1

        patch = self.data_info[i]
        
        wsi = openslide.open_slide(patch['wsi_path'])

        # get first image with lower resolution
        img_2 = wsi.read_region(patch['coords'], mag_2, (patch['patch_size'], patch['patch_size'])).convert('RGB')

        # sample coordinate inside image with lower resolution
        x = patch['coords'][0] + (patch['patch_size'] * level_downsamples[str(mag_2)])
        y = patch['coords'][1] + (patch['patch_size'] * level_downsamples[str(mag_2)])

        x_rand = random.randrange(patch['coords'][0], x - patch['patch_size'])
        y_rand = random.randrange(patch['coords'][1], y - patch['patch_size'])

        # get second image with higher resolution, mag_1
        img_1 = wsi.read_region((x_rand, y_rand), mag_1, (patch['patch_size'], patch['patch_size'])).convert('RGB')

        return img_1, img_2

