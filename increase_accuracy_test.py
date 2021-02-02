import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from datasets.Municipality.settings import cfg_data
from datasets.Municipality.loading_data import loading_data
from datasets.dataset_utils import split_image_and_den, unsplit_den, unsplit_img
import matplotlib.pyplot as plt
from matplotlib import cm as CM

from timm.models import create_model
from misc.deit_dentsity_refiner import refine_density

import custom_models


def main():
    seed = 42  # Very randomly selected. Trust me
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    model = create_model(
        'deit_small_distilled_patch16_224',
        init_path=False,
        num_classes=1000,  # Not yet used anyway. Must match pretrained model!
        drop_rate=0.,
        drop_path_rate=0.1,  # TODO: What does this do?
        drop_block_rate=None,
    )

    model = model.cuda()

    state_path = 'DAM_ROI_small_ep_700_MAE_1.327.pth'
    resume_state = torch.load(state_path)
    model.load_state_dict(resume_state['net'])

    train_loader, test_loader, restore_transform = loading_data()
    test_network(model, test_loader)


def test_network(model, test_loader):
    save_dir = 'cv_results'

    pres = []
    posts = []

    accumulated = 0
    model.eval()
    with torch.no_grad():
        # for idx, (img_patches, gt_patches, img_resolution, img) in enumerate(test_loader):
        for idx, (img_patches, gt_patches, img_resolution, img) in enumerate(test_loader):
            img_patches = img_patches.squeeze().cuda()
            gt_patches = gt_patches.squeeze()
            img_resolution = img_resolution.squeeze()
            img = img.squeeze()

            out = model(img_patches)
            out = out.squeeze().cpu()

            den, gt = unsplit_den(out, gt_patches, img_resolution)
            # refined_den = refine_density(model, den.clone(), img)

            gt_count = gt.sum() / cfg_data.LABEL_FACTOR
            pred_count_pre = den.sum() / cfg_data.LABEL_FACTOR
            # den_pre = np.array(den / cfg_data.LABEL_FACTOR)

            # pred_count_post = refined_den.sum() / cfg_data.LABEL_FACTOR
            err_pre = torch.abs(pred_count_pre - gt_count)
            # err_post = torch.abs(pred_count_post - gt_count)
            # print(f'pred: {pred_count_pre}, gt: {gt_count}')
            # accumulated += (err_pre - err_post).item()

            print(f'{idx}, pred: {pred_count_pre}, gt: {gt_count}')
            # print(f'{idx}. pre: {err_pre:.3f}, post: {err_post:.3f}, improvement: {err_pre - err_post:.3f}, acc: {accumulated:.3f}, avg acc:{accumulated / (idx + 1):.3f}')
            #
            pres.append(err_pre.item())
            # posts.append(err_post.item())

            # den_post = np.array(refined_den / cfg_data.LABEL_FACTOR)

            # plt.imshow(den_pre, cmap=CM.jet)
            # plt.title(f'(pre) {idx}, pred: {pred_count_pre:.3f}, gt: {gt_count:.1f}, err: {pres[-1]:.3f}')
            # pred_den_path = os.path.join(save_dir, f'{idx}_pre.jpg')
            # plt.savefig(pred_den_path)
            # #
            # plt.imshow(den_post, cmap=CM.jet)
            # plt.title(f'(post) {idx}, pred: {pred_count_post:.3f}, gt: {gt_count:.1f}, err: {posts[-1]:.3f}, impr: {pres[-1] - posts[-1]:.3f}')
            # pred_den_path = os.path.join(save_dir, f'{idx}_post.jpg')
            # plt.savefig(pred_den_path)
            #
            # plt.imshow(np.array(gt / cfg_data.LABEL_FACTOR), cmap=CM.jet)
            # plt.title(f'(gt) {idx}, gt: {gt_count:.1f}')
            # pred_den_path = os.path.join(save_dir, f'{idx}_gt.jpg')
            # plt.savefig(pred_den_path)

        # print(f'pre MAE: {np.mean(pres)}, post MAE: {np.mean(posts)}')
        print(f'pre MAE: {np.mean(pres)}')



if __name__ == '__main__':
    main()
    # view_preds()
