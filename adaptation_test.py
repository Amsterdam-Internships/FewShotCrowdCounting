# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import os

from datasets.WE_MAML.loading_data import loading_data

from datasets.dataset_utils import img_equal_unsplit

import matplotlib.pyplot as plt

# from CSRNet import CSRNet
import models  # Need to register the models!
import models_functional
from timm.models import create_model

save_dir = 'adapt_results_deit2'

train_loader, test_loaders, restore_transform = loading_data(224)

def adapt_to_scene(model, crit, optim, dataloader, adapt_img_idxs):
    model.train()
    loss = torch.tensor(0).float().cuda()
    optim.zero_grad()

    for idx, (img, gt) in enumerate(dataloader):
        if idx in adapt_img_idxs:
            img = img.squeeze().cuda()
            gt = gt.cuda().squeeze()

            pred = model(img).squeeze()
            loss += crit(pred, gt)
    loss = loss / len(adapt_img_idxs)
    loss.backward()
    optim.step()

    return model

def eval_on_scene(model, dataloader, adapt_img_idxs):
    model.eval()

    MAE_ = 0
    preds = []
    gts = []
    img_idxs = []

    with torch.no_grad():
        for idx, (img, gt) in enumerate(dataloader):
            if idx not in adapt_img_idxs:
                img = img.squeeze().cuda()
                gt = gt.squeeze().unsqueeze(1)

                pred = model(img)

                pred = img_equal_unsplit(pred.cpu(), 112, 4, 576, 720, 1).squeeze()
                gt = img_equal_unsplit(gt, 112, 4, 576, 720, 1).squeeze()

                pred_cnt = pred.sum() / 1000
                gt_cnt = gt.sum() / 1000
                MAE_ += abs(pred_cnt - gt_cnt)

                preds.append(pred_cnt.item())
                gts.append(gt_cnt.item())
                img_idxs.append(idx)

    MAE = MAE_ / (len(dataloader.dataset) - len(adapt_img_idxs))

    return preds, gts, img_idxs, MAE



def main():



    # adaptation_idx = [50]
    #
    # for scene_idx in range(5):
    #     print(f'running scene {scene_idx}')
    #     test_loader = test_loaders[scene_idx]
    #
    #     model = create_model(
    #         'deit_tiny_patch16_224',
    #         init_path=None,
    #         num_classes=1000,  # Not yet used anyway. Must match pretrained model!
    #         drop_rate=0.,
    #         drop_path_rate=0.1,  # TODO: What does this do?
    #         drop_block_rate=None,
    #     )
    #     model.cuda()
    #
    #     resume_state = torch.load('DeitTiny_MAML_ep_25000.pth')
    #     model.load_state_dict(resume_state['net'])
    #
    #     criterion = torch.nn.MSELoss()
    #     optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    #
    #     preds_before, gts, img_idxs, MAE_before = eval_on_scene(model, test_loader, adaptation_idx)
    #     model = adapt_to_scene(model, criterion, optimizer, test_loader, adaptation_idx)
    #     preds_after, gts, img_idxs, MAE_after = eval_on_scene(model, test_loader, adaptation_idx)
    #
    #     save_path = os.path.join(save_dir, f'scene_{scene_idx}.jpg')
    #     plt.figure()
    #     plt.plot(img_idxs, gts, 'g-', label='gt')
    #     plt.plot(img_idxs, preds_before, 'b-', label='before')
    #     plt.plot(img_idxs, preds_after, 'r--', label='after')
    #     plt.title(f'MAE from {MAE_before:.3f} to {MAE_after:.3f}')
    #     plt.savefig(save_path)
    #     print(f'MAE from {MAE_before:.3f} to {MAE_after:.3f}')
    #
    # print('done')

    scene_idx = 3
    adaptation_idx = [40, 80]
    print(f'running scene {scene_idx}')
    test_loader = test_loaders[scene_idx]

    model = create_model(
        'deit_tiny_patch16_224',
        init_path=None,
        num_classes=1000,  # Not yet used anyway. Must match pretrained model!
        drop_rate=0.,
        drop_path_rate=0.1,  # TODO: What does this do?
        drop_block_rate=None,
    )
    model.cuda()

    resume_state = torch.load('DeitTiny_MAML_ep_25000.pth')
    model.load_state_dict(resume_state['net'])

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    preds_before, gts, img_idxs, MAE_before = eval_on_scene(model, test_loader, adaptation_idx)
    model = adapt_to_scene(model, criterion, optimizer, test_loader, adaptation_idx)
    preds_after, gts, img_idxs, MAE_after = eval_on_scene(model, test_loader, adaptation_idx)

    save_path = os.path.join(save_dir, f'scene_{scene_idx}.jpg')
    plt.figure()
    plt.plot(img_idxs, gts, 'g-', label='gt')
    plt.plot(img_idxs, preds_before, 'b-', label='before')
    plt.plot(img_idxs, preds_after, 'r--', label='after')
    plt.title(f'MAE from {MAE_before:.3f} to {MAE_after:.3f}')
    # plt.savefig(save_path)
    plt.show()
    print(f'MAE from {MAE_before:.3f} to {MAE_after:.3f}')

    print('done')


# print(f'scene {ldr_idx}, before MAE: {MAE:.3f}')
#     print(f'overall MAE: {np.mean(MAEs_):.3f}')


if __name__ == '__main__':
    main()
