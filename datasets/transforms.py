import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
import torchvision.transforms as standard_transforms
from skimage import exposure, img_as_float, img_as_ubyte


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, bbx=None):
        if bbx is None:
            for t in self.transforms:
                img, mask = t(img, mask)
            return img, mask
        for t in self.transforms:
            img, mask, bbx = t(img, mask, bbx)
        return img, mask, bbx


class PILToTensor(object):
    def __call__(self, img):
        return standard_transforms.ToTensor()(np.array(img))


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, bbx=None):
        if random.random() < 0.5:
            if bbx is None:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            w, h = img.size
            xmin = w - bbx[:, 3]
            xmax = w - bbx[:, 1]
            bbx[:, 1] = xmin
            bbx[:, 3] = xmax
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), bbx
        if bbx is None:
            return img, mask
        return img, mask, bbx


class RandomCrop(object):
    def __init__(self, crop_shape):
        self.crop_w = crop_shape[0]
        self.crop_h = crop_shape[1]

    def __call__(self, img, mask, bbx=None):
        assert img.size == mask.size

        w, h = img.size
        x1 = random.randint(0, w - self.crop_w)
        y1 = random.randint(0, h - self.crop_h)

        crop_box = (x1, y1, x1 + self.crop_w, y1 + self.crop_h)
        return img.crop(crop_box), mask.crop(crop_box)


class RandomGrayscale(object):
    def __call__(self, img):
        if random.random() < 0.1:
            return img.convert('L').convert('RGB')
        else:
            return img


class RandomGammaTransform(object):
    def __call__(self, img):
        if random.random() < 0.25:
            gamma = random.uniform(0.5, 1.5)
            img = img_as_float(img)
            img = exposure.adjust_gamma(img, gamma)
            img = Image.fromarray(img_as_ubyte(img))
        return img


class LabelScale(object):
    def __init__(self, label_factor):
        self.label_factor = label_factor

    def __call__(self, tensor):
        tensor = torch.from_numpy(np.array(tensor))
        tensor = tensor * self.label_factor
        return tensor


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
