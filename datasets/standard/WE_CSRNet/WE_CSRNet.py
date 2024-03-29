import os

import numpy as np
import pandas as pd

import torch
from torch.utils import data

from PIL import Image
from .settings import cfg_data
from datasets.dataset_utils import img_equal_split, img_equal_unsplit


class WE_CSRNet(data.Dataset):
    def __init__(self, data_path, mode,
                 main_transform=None, img_transform=None, gt_transform=None):

        self.data_path = os.path.join(data_path, mode)
        self.mode = mode  # train, val or test

        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.main_transform = main_transform

        self.img_extension = '.jpg'

        self.data_files = []
        for scene in os.listdir(self.data_path):
            scene_path = os.path.join(self.data_path, scene)
            scene_imgs = [os.path.join(scene_path, 'img', img_name)
                           for img_name in os.listdir(os.path.join(scene_path, 'img'))
                           if img_name.endswith(self.img_extension)]
            self.data_files += scene_imgs  # Concatenate lists

        self.num_samples = len(self.data_files)
        print(f'{len(self.data_files)} {self.mode} images found.')

    def __getitem__(self, index):
        img_path = self.data_files[index]
        img, den = self.read_image_and_gt(img_path)

        if self.main_transform is not None:
            img, den = self.main_transform(img, den)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)

        return img, den

    def read_image_and_gt(self, img_path):
        den_path = img_path.replace('img', 'den').replace(self.img_extension, '.csv')

        img = Image.open(img_path)
        if img.mode == 'L':
            img = img.convert('RGB')

        den = pd.read_csv(den_path, header=None).values
        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)

        return img, den

    def __len__(self):
        return self.num_samples
