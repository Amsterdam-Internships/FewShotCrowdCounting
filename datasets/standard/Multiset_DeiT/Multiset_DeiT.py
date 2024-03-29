import os

import numpy as np
import pandas as pd

import torch
from torch.utils import data

from PIL import Image
from .settings import cfg_data
from datasets.dataset_utils import img_equal_split, img_equal_unsplit


class Multiset_DeiT(data.Dataset):
    def __init__(self, data_paths, mode, crop_size,
                 main_transform=None, img_transform=None, gt_transform=None, cropper=None):

        self.crop_size = crop_size  # 224
        self.mode = mode  # train, test or eval

        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.cropper = cropper

        self.data_files = []

        for data_path in data_paths:  # For each dataset
            data_path = os.path.join(data_path, mode)  # dataset/train, dataset/val, or dataset/test
            imgs_path = os.path.join(data_path, 'img')  # folder that has all the images
            self.data_files += [(data_path, img_name) for img_name in os.listdir(imgs_path) if
                                img_name.endswith('.jpg')]

        if not self.data_files:  # If we only have a train or test set, we can still initialize the dataloader.
            self.data_files = ['Dummy']  # Handy for testing on a separate test set that doesn't have a train set.
        self.num_samples = len(self.data_files)

        if self.data_files[0] == 'Dummy':
            print(f'No {self.mode} images found in {len(data_paths)} datasets.')
        else:
            print(f'{len(self.data_files)} {self.mode} images found in {len(data_paths)} datasets.')

    def __getitem__(self, index):
        """ Get img and gt stored at index 'index' in data files. """
        img, den = self.read_image_and_gt(index)

        if self.main_transform is not None:
            img, den = self.main_transform(img, den)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)

        if self.mode == 'train':
            img_crop, den_crop = self.cropper(img, den.unsqueeze(0))
            return img_crop, den_crop
        else:
            img_stack = img_equal_split(img, self.crop_size, cfg_data.OVERLAP)
            gts_stack = img_equal_split(den.unsqueeze(0), self.crop_size, cfg_data.OVERLAP)
            return img, img_stack, gts_stack

    def read_image_and_gt(self, index):
        """
        Retrieves the image and density map from the disk.
        :param index: Index of data_files.
        :return: image and gt density map as PIL Images
        """

        data_path, img_name = self.data_files[index]
        img_path = os.path.join(data_path, 'img', img_name)
        den_path = os.path.join(data_path, 'den', img_name.replace('img_', 'den_').replace('.jpg', '.csv'))

        img = Image.open(img_path)
        if img.mode == 'L':  # Black and white
            img = img.convert('RGB')  # Colour

        den = pd.read_csv(den_path, header=None).values
        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)

        return img, den

    def __len__(self):
        """ The number of paths stored in data files. """
        return self.num_samples