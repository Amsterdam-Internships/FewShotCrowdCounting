import os

import numpy as np
import pandas as pd

from PIL import Image
import csv

import scipy.ndimage



data_path = ''
img_file_extension = '.png'
data_files = [os.path.join(data_path, filename) for filename in os.listdir(data_path)
              if filename.endswith(img_file_extension)]


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
