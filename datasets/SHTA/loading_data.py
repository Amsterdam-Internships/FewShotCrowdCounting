import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import datasets.transforms as own_transforms

from .settings import cfg_data
from .SHTA import SHTA


def loading_data(crop_size):
    # Train transforms
    train_main_transform = own_transforms.Compose([
        own_transforms.RandomCrop([crop_size, crop_size]),
        own_transforms.RandomHorizontallyFlip()
    ])

    train_img_transform = standard_transforms.Compose([
        own_transforms.RandomGammaTransform(),
        own_transforms.RandomGrayscale(),
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD)
    ])

    train_gt_transform = standard_transforms.Compose([
        own_transforms.LabelScale(cfg_data.LABEL_FACTOR)
    ])

    # Test transforms
    test_main_transform = None

    test_image_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD)
    ])

    test_gt_transform = standard_transforms.Compose([
        own_transforms.LabelScale(cfg_data.LABEL_FACTOR)
    ])

    # Restore transform
    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*cfg_data.MEAN_STD),
        standard_transforms.ToPILImage()
    ])

    train_set = SHTA(cfg_data.DATA_PATH + '/train', 'train', crop_size,
                     main_transform=train_main_transform,
                     img_transform=train_img_transform,
                     gt_transform=train_gt_transform)
    train_loader = DataLoader(train_set,
                              batch_size=cfg_data.TRAIN_BS,
                              num_workers=cfg_data.N_WORKERS,
                              shuffle=True, drop_last=True)

    test_set = SHTA(cfg_data.DATA_PATH + '/test', 'test', crop_size,
                    main_transform=test_main_transform,
                    img_transform=test_image_transform,
                    gt_transform=test_gt_transform)
    test_loader = DataLoader(test_set,
                             batch_size=cfg_data.TEST_BS,
                             num_workers=cfg_data.N_WORKERS,
                             shuffle=False, drop_last=False)

    return train_loader, test_loader, restore_transform
