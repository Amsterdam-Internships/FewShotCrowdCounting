import os

import numpy as np
import pandas as pd

import torch
from torch.utils import data

from PIL import Image
from .settings import cfg_data
from datasets.dataset_utils import split_image_and_den, unsplit_den


class SHTB_train(data.Dataset):
    def __init__(self, data_path, main_transform=None, img_transform=None, gt_transform=None):
        self.img_path = data_path + '/img'
        self.gt_path = data_path + '/den'
        self.data_files = [filename for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path, filename))]
        self.num_samples = len(self.data_files)
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

    def __getitem__(self, index):
        fname = self.data_files[index]
        img, den = self.read_image_and_gt(fname)
        if self.main_transform is not None:
            img, den = self.main_transform(img, den)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)
        return img, den

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, fname):
        img = Image.open(os.path.join(self.img_path, fname))
        if img.mode == 'L':
            img = img.convert('RGB')

        den = pd.read_csv(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.csv'), sep=',', header=None).values

        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)

        return img, den

    def get_num_samples(self):
        return self.num_samples


class SHTB_test(data.Dataset):
    # For testing, split image into pieces such that the pieces make up the whole.
    # Batch size must be 1
    def __init__(self, data_path, main_transform=None, img_transform=None, gt_transform=None):
        self.img_path = data_path + '/img'
        self.gt_path = data_path + '/den'
        self.data_files = [filename for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path, filename))]
        self.num_samples = len(self.data_files)
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

    def __getitem__(self, index):
        fname = self.data_files[index]
        img, den = self.read_image_and_gt(fname)
        imgs, dens = split_image_and_den(img, den, 224)  # TODO: put 224 in model for larget model support
        for i in range(len(imgs)):
            if self.img_transform:  # These should always be provided
                imgs[i] = self.img_transform(imgs[i])
            if self.gt_transform:
                dens[i] = self.gt_transform(dens[i])
        return torch.stack(imgs), torch.stack(dens), torch.tensor(img.size)

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, fname):
        img = Image.open(os.path.join(self.img_path, fname))
        if img.mode == 'L':
            img = img.convert('RGB')

        den = pd.read_csv(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.csv'), sep=',', header=None).values

        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)

        return img, den

    def get_num_samples(self):
        return self.num_samples
