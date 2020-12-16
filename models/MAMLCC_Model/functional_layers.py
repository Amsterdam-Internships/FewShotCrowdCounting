import torch
from torch import nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return F.conv2d(input, weight.to(device), bias.to(device), stride, padding, dilation, groups)


def relu(input):
    return F.threshold(input, 0, 0, inplace=True)


def maxpool(input, kernel_size, stride=None):
    return F.max_pool2d(input, kernel_size, stride)


def bilinear_upsample(in_, factor):
    return F.upsample(in_, None, factor, 'bilinear')


def log_softmax(input):
    return F.log_softmax(input)
