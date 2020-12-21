import numpy as np
import os
import random
from scipy import io as sio
import sys
import torch
from torch.utils import data
from PIL import Image, ImageOps

import pandas as pd

def read_image_and_gt(img_path):
    gt_path = img_path.replace('.jpg', '.csv').replace('frames', 'csvs')

    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')

    den = pd.read_csv(gt_path, sep=',', header=None).values
    den = den.astype(np.float32, copy=False)
    den = Image.fromarray(den)

    return img, den


class WEAC(data.Dataset):
    def __init__(self, data_path, n_unlabeled=1, n_labeled=3, main_transform=None, img_transform=None, gt_transform=None):
        self.train_path = os.path.join(data_path, 'frames')
        self.frame_paths = [os.path.join(self.train_path, frame) for frame in os.listdir(self.train_path)]

        self.n_unlabeled = n_unlabeled
        self.n_labeled = n_labeled

        self.batches = self.prepare_all_batches()

        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

    def prepare_all_batches(self):
        select_size = self.n_unlabeled + self.n_labeled


        training_batches = []
        random.shuffle(self.frame_paths)  # Because changes in place
        for frame_path in self.frame_paths:
            imgs_paths = os.listdir(frame_path)
            for i in range(len(imgs_paths)):
                img_path = imgs_paths[i]
                imgs_paths[i] = os.path.join(frame_path, img_path)
            random.shuffle(imgs_paths)  # Because changes in place

            n_imgs = len(imgs_paths)
            cur_idx = 0
            while cur_idx + select_size < n_imgs:
                training_sample = imgs_paths[cur_idx:cur_idx + select_size]
                training_batches.append(training_sample)  # Note that we use append and not extend
                cur_idx += select_size

        return training_batches

    def __getitem__(self, index):
        batch = self.batches[index]
        imgs = []
        gts = []
        for image in batch:
            img, den = read_image_and_gt(image)
            if self.main_transform is not None:
                img, den = self.main_transform(img, den)
            if self.img_transform is not None:
                img = self.img_transform(img)
            if self.gt_transform is not None:
                den = self.gt_transform(den)
            imgs.append(img)
            gts.append(den)

        unlabled_imgs = imgs[:self.n_unlabeled]
        labeled_imgs = imgs[self.n_unlabeled:self.n_unlabeled + self.n_labeled]
        labeled_gts = gts[self.n_unlabeled:self.n_unlabeled + self.n_labeled]

        return unlabled_imgs, labeled_imgs, labeled_gts

    def __len__(self):
        return len(self.batches)

    def get_num_samples(self):  # TODO: Is this even used?
        return len(self)
