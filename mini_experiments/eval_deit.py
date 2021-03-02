import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import os

import models.DeiTModels  # Need to register the models!
from timm.models import create_model

from datasets.dataset_utils import img_equal_unsplit
from datasets.SHTB_DeiT.loading_data import loading_data


def evaluate_model(model, dataloader):

    model.eval()
    with torch.no_grad():
        AEs = []  # Absolute Errors
        SEs = []  # Squared Errors


        for idx, (img, img_patches, gt_patches) in enumerate(dataloader):
            img_patches = img_patches.squeeze().cuda()
            gt_patches = gt_patches.squeeze().unsqueeze(1)  # Remove batch dim, insert channel dim
            img = img.squeeze()  # Remove batch dimension
            _, img_h, img_w = img.shape

            pred_den = model(img_patches)
            pred_den = pred_den.cpu()

            gt = img_equal_unsplit(gt_patches, 8, 4, img_h, img_w, 1)
            den = img_equal_unsplit(pred_den, 8, 4, img_h, img_w, 1)
            den = den.squeeze()  # Remove channel dim

            pred_cnt = den.sum() / 1000
            gt_cnt = gt.sum() / 1000
            AEs.append(torch.abs(pred_cnt - gt_cnt).item())
            SEs.append(torch.square(pred_cnt - gt_cnt).item())
            print(f'pred: {pred_cnt:.3f}, gt: {gt_cnt:.3f}, AE: {AEs[-1]:.3f}, SE: {SEs[-1]:.3f}')

        MAE = np.mean(AEs)
        MSE = np.sqrt(np.mean(SEs))

    return MAE, MSE

def main():
    _, test_loader, _ = loading_data(224)

    model = create_model(
        'deit_small_distilled_patch16_224',
        init_path=None,
        num_classes=1000,  # Not yet used anyway. Must match pretrained model!
        drop_rate=0.,
        drop_path_rate=0.1,  # TODO: What does this do?
        drop_block_rate=None,
    )

    model.cuda()

    resume_state = torch.load('save_state_ep_430_new_best_MAE_9.058.pth')
    model.load_state_dict(resume_state['net'])

    MAE, MSE = evaluate_model(model, test_loader)
    print(MAE, MSE)


if __name__ == '__main__':
    main()