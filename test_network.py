import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from datasets.Municipality.settings import cfg_data
from datasets.Municipality.loading_data import loading_data
from datasets.dataset_utils import split_image_and_den, unsplit_den, unsplit_img
import matplotlib.pyplot as plt

import custom_models
from timm.models import create_model
import ntpath
import csv
from PIL import Image
import pandas as pd

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

    state_path = 'runs/01-28_09-11/STATE_DICTS/save_state_ep_475_new_best_MAE_1.891.pth'
    resume_state = torch.load(state_path)
    model.load_state_dict(resume_state['net'])

    train_loader, test_loader, restore_transform = loading_data()
    test_network(model, test_loader)


def test_network(model, test_loader):


    save_dir = 'cv_results'

    errors = []
    gt_counts = []
    pred_counts = []
    img_names = []
    overall_summed = 0
    model_summed = 0
    no_gpu_summed = 0
    unsplit_summed = 0

    model.eval()
    with torch.no_grad():
        for idx, (img_patches, gt_patches, img_resolution) in enumerate(test_loader):
            overall_time_start = time.time()

            img_patches = img_patches.squeeze().cuda()
            gt_patches = gt_patches.squeeze()
            img_resolution = img_resolution.squeeze()
            img_path = test_loader.dataset.data_files[idx]
            img_name = ntpath.split(img_path)[-1]

            model_time_start = time.time()
            no_gpu_time_start = time.time()
            out = model(img_patches)
            model_summed += time.time() - model_time_start

            out = out.squeeze().cpu()

            unsplit_start = time.time()
            den, gt = unsplit_den(out, gt_patches, img_resolution)
            unsplit_summed += time.time() - unsplit_start

            pred_cnt = den.sum() / cfg_data.LABEL_FACTOR
            gt_cnt = torch.round(gt.sum() / cfg_data.LABEL_FACTOR)  # Can be float if den outside image
            errors.append(torch.abs(pred_cnt - gt_cnt))

            overall_summed += time.time() - overall_time_start
            no_gpu_summed += time.time() - no_gpu_time_start

            img_names.append(img_name)
            pred_counts.append(pred_cnt)
            gt_counts.append(gt_cnt)


            den = np.array(den)

            pred_den_name = img_name.replace('.jpg', 'pred-den.csv')
            pred_den_path = os.path.join(save_dir, pred_den_name)
            with open(pred_den_path, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                for line in den:
                    writer.writerow(line)

    for idx in torch.argsort(torch.stack(gt_counts)):
        print(f'img: {img_names[idx]}. Predicted count: {pred_counts[idx]:.3f}, gt count: {int(gt_counts[idx])}')

    for idx in torch.argsort(torch.stack(gt_counts)):
        print(f'{pred_counts[idx]:.3f} {int(gt_counts[idx])}')

    MAE = torch.mean(torch.stack(errors))
    print(f'MAE: {MAE:.3f}')
    avg_total_time = overall_summed / len(test_loader.dataset)
    avg_forward_time = model_summed / len(test_loader.dataset)
    avg_no_gpu_time = no_gpu_summed / len(test_loader.dataset)
    avg_unsplit_time = unsplit_summed / len(test_loader.dataset)

    print(f'AVG total time: {avg_total_time:.3f}, '
          f'AVG forward time: {avg_forward_time:.3f}, '
          f'AVG no GPU time: {avg_no_gpu_time:.3f}, '
          f'AVG unsplit time: {avg_unsplit_time:.3f}')
    print(f'Overall img per sec: {1 / avg_total_time:.3f}')
    print(f'Overall forward per sec: {1 / avg_forward_time:.3f}')
    print(f'Overall no GPU img per sec: {1 / avg_no_gpu_time:.3f}')


def view_preds():
    path_to_saves = 'cv_results'
    path_to_dataset = 'actual_data_datasets\\MT_Picnic_CV_ManualSelection\\val'

    imgs_to_show = [
        'img_5288c9a5-4ff0-497c-9876-afb7c24a0cb5',
        'img_14db3fb4-3fd7-4373-9a38-d4f4db5a6f32',
        'img_3e9c6c81-9dad-4974-b2fb-d24badb9f844',
        'img_21b8ec47-ac31-4ee7-92b4-2aa55b4db508'
    ]



    for img_to_show in imgs_to_show:
        img_path = os.path.join(path_to_dataset, img_to_show + '.jpg')
        gt_path = os.path.join(path_to_dataset, img_to_show + '-den.csv')
        pred_den_path = os.path.join(path_to_saves, img_to_show + 'pred-den.csv')

        img = Image.open(img_path)
        if img.mode == 'L':
            img = img.convert('RGB')
        img = np.asarray(img)

        gt_den = pd.read_csv(gt_path, header=None).values
        gt_den = gt_den.astype(np.float32, copy=False)

        pred_den = pd.read_csv(pred_den_path, header=None).values
        pred_den = pred_den.astype(np.float32, copy=False) / 1000

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
        axes[0].imshow(img)
        axes[1].imshow(gt_den)
        axes[1].set_title(f'gt count: {np.round(gt_den.sum())}')
        axes[2].imshow(pred_den)
        axes[2].set_title(f'pred count: {np.sum(pred_den):.3f}')
        fig.tight_layout()
        plt.show()




if __name__ == '__main__':
    main()
    # view_preds()

