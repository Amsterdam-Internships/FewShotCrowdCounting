import os

import numpy as np
import pandas as pd

import torch
from torch.utils import data

from PIL import Image
from .settings import cfg_data
from datasets.dataset_utils import split_image_and_den, unsplit_den
from datasets.dataset_utils import generate_density_municipality



class Muni(data.Dataset):
    def __init__(self, data_path, mode, crop_size, main_transform=None, img_transform=None, gt_transform=None):
        self.data_path = data_path
        self.crop_size = crop_size
        self.mode = mode  # train or test

        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

        self.img_extension = ''
        for file in os.listdir(data_path):
            if file.endswith('.jpg'):
                self.img_extension = '.jpg'
            elif file.endswith('.png'):
                self.img_extension = '.png'
        if not self.img_extension:
            print(f'Could not find proper file extension for the images in {data_path}.')

        self.data_files = [os.path.join(data_path, file) for file in os.listdir(data_path)
                           if file.endswith(self.img_extension)]

        if not self.data_files:          # If we only have a train or test set, we can still initialize the dataloader.
            self.data_files = ['Dummy']  # Handy for testing on a separate test set that doesn't have a train set.
        self.num_samples = len(self.data_files)

        print(f'{len(self.data_files)} {self.mode} images found.')

    def __getitem__(self, index):
        img_path = self.data_files[index]
        img, den = self.read_image_and_gt(img_path)
        if self.mode == 'train':
            return self.process_getitem_train(img, den)
        else:
            return self.process_getitem_test(img, den)

    def read_image_and_gt(self, img_path):
        den_path = img_path.replace(self.img_extension, '-den.csv')

        img = Image.open(img_path)
        if img.mode == 'L':
            img = img.convert('RGB')

        den = pd.read_csv(den_path, header=None).values
        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)

        return img, den

    def process_getitem_train(self, img, den):
        if self.main_transform is not None:
            img, den = self.main_transform(img, den)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)
        return img, den

    def process_getitem_test(self, img, den):
        imgs, dens = split_image_and_den(img, den, self.crop_size)
        for i in range(len(imgs)):
            if self.img_transform:  # These should always be provided
                imgs[i] = self.img_transform(imgs[i])
            if self.gt_transform:
                dens[i] = self.gt_transform(dens[i])
        return torch.stack(imgs), torch.stack(dens), torch.tensor(img.size)

    def __len__(self):
        return self.num_samples
