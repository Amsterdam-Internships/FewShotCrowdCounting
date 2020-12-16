import numpy as np
from torch.utils import data
from PIL import Image, ImageOps
import random
import cv2
import os
from .setting import cfg_data
from torchvision import transforms


def read_image_and_gt(img_path):
    gt_path = img_path.replace('.jpg', '.csv').replace('frames', 'csvs')

    img = Image.open(img_path).convert('RGB')
    target = np.loadtxt(gt_path, delimiter=',')

    return img, target


class WEMAML(data.Dataset):
    def __init__(self, data_files, mode='train', img_transform=None, gt_transform=None):
        self.data_files = data_files
        self.num_samples = len(self.data_files)
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.mode = mode

    def __getitem__(self, index):
        img_path = self.data_files[index]
        img, target = read_image_and_gt(img_path)

        if self.mode == 'train':
            if random.random() > 0.8:
                target = np.fliplr(target)
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w, h = target.shape
            new_w = (w // 8) * 8
            new_h = (h // 8) * 8
            # target = cv2.resize(target, (new_h, new_w), interpolation=cv2.INTER_CUBIC)
            target = cv2.resize(target, (new_h, new_w))

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.gt_transform is not None:
            target = self.gt_transform(target)

        return img, target

    def __len__(self):
        return self.num_samples

    def get_num_samples(self):
        return self.num_samples
