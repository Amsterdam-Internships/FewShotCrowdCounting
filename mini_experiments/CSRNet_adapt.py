# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import os

from datasets.WE_MAML.loading_data import loading_data

from collections import OrderedDict

import matplotlib.pyplot as plt

from models.CSRNet import CSRNet
from models.CSRNet_functional import CSRNet_functional

save_dir = 'adapt_CSRNet_1000'

train_loader, test_loaders, restore_transform = loading_data(224)

def adapt_to_scene(model_functional, theta, crit, dataloader, adapt_img_idxs):

    theta_weights = list(theta[k] for k in theta if not k.startswith('alpha.'))
    theta_names = list(k for k in theta if not k.startswith('alpha.'))
    alpha_weights = list(theta[k] for k in theta if k.startswith('alpha.'))

    imgs = []
    gts = []
    for idx in adapt_img_idxs:
        img, gt = dataloader.dataset.__getitem__(idx)
        img = img.cuda()
        gt = gt.cuda()
        imgs.append(img)
        gts.append(gt)

    img_stack = torch.stack(imgs)
    gt_stack = torch.stack(gts)
    pred = model_functional(img_stack, theta).squeeze()
    loss = crit(pred, gt_stack)

    grads = torch.autograd.grad(loss, theta_weights)
    for i, n in enumerate(theta_names):
        theta[n] = theta_weights[i] - alpha_weights[i] * 2 * grads[i]

    # theta_prime = OrderedDict((n, w - a * g) for n, w, a, g in zip(theta_names, theta_weights, alpha_weights, grads))

    return theta

def eval_on_scene(model_functional, theta_prime, dataloader, adapt_img_idxs):
    MAE_ = 0
    preds = []
    gts = []
    img_idxs = []

    with torch.no_grad():
        for idx, (img, gt) in enumerate(dataloader):
            if idx not in adapt_img_idxs:
                img = img.cuda()

                pred = model_functional(img, theta_prime)

                pred_cnt = pred.sum() / 1000
                gt_cnt = gt.sum() / 1000
                MAE_ += abs(pred_cnt - gt_cnt)

                preds.append(pred_cnt.item())
                gts.append(gt_cnt.item())
                img_idxs.append(idx)

    MAE = MAE_ / (len(dataloader.dataset) - len(adapt_img_idxs))

    return preds, gts, img_idxs, MAE


def analyse_alpha(model):
    alpha = model.alpha
    values = []
    for k in alpha.keys():
        v = alpha[k]
        vm = v.mean().item()
        vam = v.abs().mean().item()
        vstd = v.std().item()
        print(f'{k}, mean: {vm:.3f}, abs mean: {vam:.3f}, std dev: {vstd:.3f}')


# def before_after(model, model_functional, criterion, dataloader):
    # check_images = [20, 40, 60]
    # train_images = [40]
    #
    # for img_idx in check_images:
    #     img, gt = dataloader.dataset.__getattr__
    #

def main():



    # adaptation_idx = [50]
    #
    # for scene_idx in range(5):
    #     print(f'running scene {scene_idx}')
    #     test_loader = test_loaders[scene_idx]
    #
    #     model = CSRNet().cuda()
    #     model_functional = CSRNet_functional()
    #
    #     resume_state = torch.load('save_state_ep_24000.pth')
    #     model.load_state_dict(resume_state['net'])
    #
    #     criterion = torch.nn.MSELoss()
    #
    #     theta = OrderedDict((name, param) for name, param in model.named_parameters())
    #
    #     preds_before, gts, img_idxs, MAE_before = eval_on_scene(model_functional, theta, test_loader, adaptation_idx)
    #     theta_prime = adapt_to_scene(model, criterion, test_loader, adaptation_idx)
    #     preds_after, gts, img_idxs, MAE_after = eval_on_scene(model_functional, theta_prime, test_loader, adaptation_idx)
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

    scene_idx = 1
    adaptation_idx = [40, 80, 100]
    print(f'running scene {scene_idx}')
    test_loader = test_loaders[scene_idx]

    model = CSRNet().cuda()
    model_functional = CSRNet_functional()

    resume_state = torch.load('save_state_ep_34000.pth')
    model.load_state_dict(resume_state['net'])

    criterion = torch.nn.MSELoss()

    analyse_alpha(model)

    theta = OrderedDict((name, param) for name, param in model.named_parameters())

    preds_before, gts, img_idxs, MAE_before = eval_on_scene(model_functional, theta, test_loader, adaptation_idx)
    for _ in range(2):
        theta = adapt_to_scene(model_functional, theta, criterion, test_loader, adaptation_idx)
        torch.cuda.empty_cache()
    preds_after, gts, img_idxs, MAE_after = eval_on_scene(model_functional, theta, test_loader, adaptation_idx)

    save_path = os.path.join(save_dir, f'scene_{scene_idx}.jpg')
    plt.figure()
    plt.plot(img_idxs, gts, 'g-', label='gt')
    plt.plot(img_idxs, preds_before, 'b-', label='before')
    plt.plot(img_idxs, preds_after, 'r--', label='after')
    plt.legend()
    plt.title(f'MAE from {MAE_before:.3f} to {MAE_after:.3f}')
    plt.xlabel('Image')
    plt.ylabel('Count')
    # plt.savefig(save_path)
    plt.show()
    print(f'MAE from {MAE_before:.3f} to {MAE_after:.3f}')

    print('done')


# print(f'scene {ldr_idx}, before MAE: {MAE:.3f}')
#     print(f'overall MAE: {np.mean(MAEs_):.3f}')


if __name__ == '__main__':
    main()
