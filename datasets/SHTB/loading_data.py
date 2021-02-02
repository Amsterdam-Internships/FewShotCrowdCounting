import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import datasets.transforms as own_transforms

from .settings import cfg_data
from .SHTB import SHTB_train, SHTB_test


def loading_data(patch_size):
    train_main_transform = own_transforms.Compose([
        own_transforms.RandomCrop([patch_size, patch_size]),
        own_transforms.RandomHorizontallyFlip()
    ])

    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD)
    ])

    gt_transform = standard_transforms.Compose([
        own_transforms.LabelScale(cfg_data.LABEL_FACTOR)
    ])

    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*cfg_data.MEAN_STD),
        standard_transforms.ToPILImage()
    ])

    # TODO: train, val, test support
    # TODO: .json support
    train_set = SHTB_train(cfg_data.DATA_PATH + '/train',
                           main_transform=train_main_transform,
                           img_transform=img_transform,
                           gt_transform=gt_transform)
    train_loader = DataLoader(train_set,
                              batch_size=cfg_data.TRAIN_BS,
                              num_workers=cfg_data.N_WORKERS,
                              shuffle=True, drop_last=True)

    test_set = SHTB_test(cfg_data.DATA_PATH + '/test',
                         main_transform=None,
                         img_transform=img_transform,
                         gt_transform=gt_transform)
    test_loader = DataLoader(test_set,
                             batch_size=cfg_data.TEST_BS,
                             num_workers=cfg_data.N_WORKERS,
                             shuffle=False, drop_last=False)

    return train_loader, test_loader, restore_transform
