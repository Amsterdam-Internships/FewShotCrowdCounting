import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import misc.transforms as own_transforms
from .SHHB import SHHB
from .setting import cfg_data
import torch


def loading_data():
    mean_std = cfg_data.MEAN_STD  # Not used
    log_para = cfg_data.LOG_PARA  # Not used

    train_main_transform = own_transforms.Compose([
        own_transforms.RandomHorizontallyFlip()
    ])

    img_transform = standard_transforms.Compose([
        own_transforms.RandomGammaTransform(),
        standard_transforms.ToTensor(),
    ])
    gt_transform = standard_transforms.Compose([
        own_transforms.LabelNormalize(log_para)
    ])
    restore_transform = standard_transforms.Compose([  # This will not bring back original image. TODO: Fix later
        standard_transforms.ToPILImage()
    ])


    val_main_transform = None
    val_img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
    ])
    val_gt_trainform = standard_transforms.Compose([
        own_transforms.PILToTensor(),
    ])

    train_set = SHHB(cfg_data.DATA_PATH + '/train', 'train',
                     main_transform=train_main_transform, img_transform=img_transform, gt_transform=gt_transform)
    train_loader = DataLoader(train_set, batch_size=cfg_data.TRAIN_BATCH_SIZE, num_workers=8, shuffle=True,
                              drop_last=True)

    val_set = SHHB(cfg_data.DATA_PATH + '/test', 'test',
                   main_transform=val_main_transform, img_transform=val_img_transform, gt_transform=val_gt_trainform)
    val_loader = DataLoader(val_set, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=8, shuffle=True, drop_last=False)

    return train_loader, val_loader, restore_transform
