import math
import torch
import scipy.ndimage
import numpy as np


def generate_density_municipality(img, gt_points, sigma):
    w, h = img.size
    k = np.zeros((h, w))

    density = None
    for (x, y, _) in gt_points.astype(int):
        if x < w and y < h:
            k[y, x] = 1  # Note the order of x and y here. Height is stored in first dimension\n",
        else:
            print("This should never happen!")  # This would mean a head is annotated outside the image.\n"
    density = scipy.ndimage.filters.gaussian_filter(k, sigma, mode='constant')
    return density


def split_image_and_den(img, den, p_size):
    w, h = img.size

    n_cols = math.ceil(w / p_size)
    n_rows = math.ceil(h / p_size)

    img_patches = []
    den_patches = []
    for row in range(n_rows):
        for col in range(n_cols):
            x1 = row * p_size if row < (n_rows - 1) else h - p_size  # Last patch might not fit in image.
            x2 = x1 + p_size
            y1 = col * p_size if col < (n_cols - 1) else w - p_size  # Hence, there might also be some overlap
            y2 = y1 + p_size

            img_patches.append(img.crop((y1, x1, y2, x2)))
            den_patches.append(den.crop((y1, x1, y2, x2)))

    return img_patches, den_patches


def unsplit_den(out_patches, gt_patches, img_resolution):
    w, h = img_resolution[0], img_resolution[1]
    p_size = out_patches.shape[1]
    n_cols = math.ceil(w / p_size)
    n_rows = math.ceil(h / p_size)

    den = torch.zeros(h, w)
    gt = torch.zeros(h, w)
    divider = torch.zeros(h, w)  # Takes care of overlap

    i = 0
    for row in range(n_rows):
        for col in range(n_cols):
            x1 = row * p_size if row < (n_rows - 1) else h - p_size
            x2 = x1 + p_size
            y1 = col * p_size if col < (n_cols - 1) else w - p_size
            y2 = y1 + p_size

            den[x1:x2, y1:y2] += out_patches[i]
            divider[x1:x2, y1:y2] += torch.ones(p_size, p_size)
            gt[x1:x2, y1:y2] = gt_patches[i]  # GT does not need to average, as each pixel is guaranteed to be correct.

            i += 1

    return den / divider, gt


def unsplit_img(img_patches, img_resolution):
    w, h = img_resolution
    p_size = img_patches.shape[2]
    n_cols = math.ceil(w / p_size)
    n_rows = math.ceil(h / p_size)

    img = torch.zeros(3, h, w)

    i = 0
    for row in range(n_rows):
        for col in range(n_cols):
            x1 = row * p_size if row < (n_rows - 1) else h - p_size
            x2 = x1 + p_size
            y1 = col * p_size if col < (n_cols - 1) else w - p_size
            y2 = y1 + p_size

            img[:, x1:x2, y1:y2] = img_patches[i]
            i += 1

    return img
