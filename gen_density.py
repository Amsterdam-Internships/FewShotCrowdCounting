import os

import numpy as np
import pandas as pd

import torch
from torch.utils import data

from PIL import Image
from datasets.dataset_utils import split_image_and_den, unsplit_den
import csv
import matplotlib.pyplot as plt

import scipy.ndimage


def generate_density_municipality(img, gt_points, sigma):
    w, h = img.size
    k = np.zeros((h, w))

    for (x, y, _) in gt_points.astype(int):
        if x < w and y < h:
            k[y, x] = 1  # Note the order of x and y here. Height is stored in first dimension\n",
        else:
            print("This should never happen!")  # This would mean a head is annotated outside the image.\n"
    density = scipy.ndimage.filters.gaussian_filter(k, sigma, mode='constant')
    return density


data_path = 'actual_data_datasets\\DAM_CrossVal_Combined-density_export'
data_files = [os.path.join(data_path, filename) for filename in os.listdir(data_path)
              if filename.endswith('.png')]


for img_path in data_files:

    gt_path = img_path.replace('.png', '-tags.csv')
    den_path = img_path.replace('.png', '-den.csv')
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')

    dots = pd.read_csv(gt_path, header=0).values
    den = generate_density_municipality(img, dots, 4)
    den = den.astype(np.float32)
    with open(den_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for line in den:
            writer.writerow(line)


# counts = []
#
# for file in data_files:
#     den = pd.read_csv(file, header=None).values
#     den = den.astype(np.float32, copy=False)
#     counts.append(np.sum(den))
#
# print(f'min: {np.min(counts)}, max: {np.max(counts):.3f}, mean: {np.mean(counts):.3f}')

