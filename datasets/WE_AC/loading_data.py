import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import misc.transforms as own_transforms
from .WEAC import WEAC
from .setting import cfg_data
import torch
import random


def loading_data():
    mean_std = cfg_data.MEAN_STD
    log_para = cfg_data.LOG_PARA

    train_main_transform = own_transforms.Compose([
        own_transforms.RandomCrop(cfg_data.TRAIN_SIZE),
        own_transforms.RandomHorizontallyFlip()
    ])
    val_main_transform = None
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    gt_transform = standard_transforms.Compose([
        own_transforms.LabelNormalize(log_para)
    ])
    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    train_set = WEAC(cfg_data.DATA_PATH + '/test/', n_unlabeled=cfg_data.N_UNLABELED, n_labeled=cfg_data.N_LABELED,
                     main_transform=train_main_transform, img_transform=img_transform, gt_transform=gt_transform)

    # Drop last should not be needed since this is already taken care of by WEAC
    train_loader = DataLoader(train_set, batch_size=1, num_workers=8, shuffle=True, drop_last=False)
    # Changing batch sizes is done through changing the number of unlabeled and labeled images.

    test_data = WEAC(cfg_data.DATA_PATH + '/test/', n_unlabeled=cfg_data.N_UNLABELED, n_labeled=cfg_data.N_LABELED,
                     main_transform=val_main_transform, img_transform=img_transform, gt_transform=gt_transform)

    val_loader = DataLoader(test_data, batch_size=1, num_workers=8, shuffle=False, drop_last=False)

    return train_loader, val_loader, restore_transform
