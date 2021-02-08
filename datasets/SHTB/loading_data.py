import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import datasets.transforms as own_transforms

from .settings import cfg_data
from .SHTB import SHTB


def loading_data(crop_size):
    # train transforms
    train_main_transform = own_transforms.Compose([
        own_transforms.RandomHorizontallyFlip()
    ])

    train_img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD)
    ])

    train_cropper = own_transforms.Compose([
        own_transforms.RandomTensorCrop([crop_size, crop_size])
    ])

    # Test transforms
    test_main_transform = None

    test_img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD)
    ])

    # Same transforms
    gt_transform = standard_transforms.Compose([
        own_transforms.LabelScale(cfg_data.LABEL_FACTOR)
    ])

    # Restore transform
    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*cfg_data.MEAN_STD),
        standard_transforms.ToPILImage()
    ])

    train_set = SHTB(cfg_data.DATA_PATH + '/train', 'train', crop_size,
                     main_transform=train_main_transform,
                     img_transform=train_img_transform,
                     gt_transform=gt_transform,
                     cropper=train_cropper)
    train_loader = DataLoader(train_set,
                              batch_size=cfg_data.TRAIN_BS,
                              num_workers=cfg_data.N_WORKERS,
                              shuffle=True, drop_last=True)

    test_set = SHTB(cfg_data.DATA_PATH + '/test', 'test', crop_size,
                    main_transform=test_main_transform,
                    img_transform=test_img_transform,
                    gt_transform=gt_transform,
                    cropper=None)
    test_loader = DataLoader(test_set,
                             batch_size=cfg_data.TEST_BS,
                             num_workers=cfg_data.N_WORKERS,
                             shuffle=False, drop_last=False)

    return train_loader, test_loader, restore_transform
