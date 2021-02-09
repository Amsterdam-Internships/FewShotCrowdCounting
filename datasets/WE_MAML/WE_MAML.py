import os
import random

import numpy as np
import pandas as pd

import torch
from torch.utils import data

from PIL import Image
from .settings import cfg_data


class WE_MAML(data.Dataset):
    def __init__(self, data_path, mode, crop_size,
                 main_transform=None, img_transform=None, gt_transform=None, splitter=None):
        self.data_path = os.path.join(data_path, 'frames')  # Only save img paths, replace with csvs when getitem
        self.crop_size = crop_size
        self.mode = mode  # train or test

        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.splitter = splitter

        self.img_extension = '.jpg'

        self.scenes = os.listdir(self.data_path)
        self.data_files = {}

        for scene in self.scenes:
            scene_dir = os.path.join(self.data_path, scene)
            self.data_files[scene] = [os.path.join(scene_dir, frame) for frame in os.listdir(scene_dir)]

        self.num_samples = len(self.scenes)

        print(f'{self.num_samples} scenes found.')

    def __getitem__(self, index):
        scene = self.scenes[index]
        n_datapoints = 1 + 1
        data_files = random.sample(self.data_files[scene], n_datapoints)
        _img_stack = []
        _gts_stack = []
        for file in data_files:
            img, den = self.read_image_and_gt(file)
            if self.main_transform is not None:
                img, den = self.main_transform(img, den)
            if self.img_transform is not None:
                img = self.img_transform(img)
            if self.gt_transform is not None:
                den = self.gt_transform(den)
            # imgs = self.splitter(img, self.crop_size, cfg_data.OVERLAP)  # Must be provided!
            # dens = self.splitter(den.unsqueeze(0), self.crop_size, cfg_data.OVERLAP)  # Must be provided!
            # _img_stack.append(imgs)
            # _gts_stack.append(dens)
            _img_stack.append(img)
            _gts_stack.append(den)
        return torch.cat(_img_stack[0:1]), torch.cat(_gts_stack[0:1]), \
               torch.cat(_img_stack[1:n_datapoints]), torch.cat(_gts_stack[1:n_datapoints])  # Meow Meow

    def read_image_and_gt(self, img_path):
        den_path = img_path.replace('frames', 'csvs').replace(self.img_extension, '.csv')

        img = Image.open(img_path)
        if img.mode == 'L':
            img = img.convert('RGB')

        den = pd.read_csv(den_path, header=None).values
        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)

        return img, den

    def __len__(self):
        return self.num_samples