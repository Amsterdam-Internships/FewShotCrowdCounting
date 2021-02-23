import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from datasets.SHTB.settings import cfg_data
from datasets.SHTB.loading_data import loading_data
from datasets.dataset_utils import img_equal_unsplit
import matplotlib.pyplot as plt
from matplotlib import cm as CM

from timm.models import create_model
from misc.deit_dentsity_refiner import refine_density

import models


def main():
    seed = 42  # Very randomly selected. Trust me
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    model = create_model(
        'deit_tiny_cnn_patch16_224',
        init_path=False,
        num_classes=1000,  # Not yet used anyway. Must match pretrained model!
        drop_rate=0.,
        drop_path_rate=0.1,  # TODO: What does this do?
        drop_block_rate=None,
    )

    model = model.cuda()

    state_path = 'Tiny_SHTB_non-lin_test/Tiny32CNN_SHTB_2ReLU_ep_1600_.pth'
    resume_state = torch.load(state_path)
    model.load_state_dict(resume_state['net'])

    train_loader, test_loader, restore_transform = loading_data(model.crop_size)
    test_network(model, test_loader)


def test_network(model, test_loader):
    save_dir = 'cv_results'

    pres = []
    posts = []

    toral_error = 0
    model.eval()
    with torch.no_grad():
        # for idx, (img_patches, gt_patches, img_resolution, img) in enumerate(test_loader):
        for idx, (img, img_patches, gt_patches) in enumerate(test_loader):
            img_patches = img_patches.squeeze().cuda()
            gt_patches = gt_patches.squeeze().unsqueeze(1)  # Remove batch dim, insert channel dim
            img = img.squeeze()  # Remove batch dimension
            _, img_h, img_w = img.shape

            pred_den, pred_count = model(img_patches)
            pred_den = pred_den.cpu()

            gt = img_equal_unsplit(gt_patches, cfg_data.OVERLAP, cfg_data.IGNORE_BUFFER, img_h, img_w, 1)
            den = img_equal_unsplit(pred_den, cfg_data.OVERLAP, cfg_data.IGNORE_BUFFER, img_h, img_w, 1)
            den = den.squeeze()  # Remove channel dim

            pred_cnt = den.sum() / 10000
            gt_cnt = gt.sum() / 10000
            toral_error += torch.abs(pred_cnt - gt_cnt)
            # print(f'pred: {pred_cnt.cpu().item():.3f}, gt: {gt_cnt.cpu().item():.3f}')


    print(toral_error / 316)

if __name__ == '__main__':
    main()
    # view_preds()
