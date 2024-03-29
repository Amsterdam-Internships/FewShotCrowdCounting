import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import datasets.transforms as own_transforms

from .settings import cfg_data
from .WE_CSRNet_Meta import WE_CSRNet_Meta, WE_CSRNet_Meta_eval


def loading_data(test_adapt_imgs=None):
    train_main_transform = own_transforms.Compose([
        own_transforms.DeterministicHorizontallyFlip()  # Either all or none of the batch images must be flipped
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

    train_set = WE_CSRNet_Meta(cfg_data.DATA_PATH, 'train',
                               main_transform=train_main_transform,
                               img_transform=img_transform,
                               gt_transform=gt_transform)
    train_loader = DataLoader(train_set,
                              batch_size=cfg_data.TRAIN_BS,
                              num_workers=cfg_data.N_WORKERS,
                              shuffle=True, drop_last=True)

    val_loaders = []
    for scene in cfg_data.VAL_SCENES:
        val_set = WE_CSRNet_Meta_eval(cfg_data.DATA_PATH, 'val', scene, n_adapt_imgs=cfg_data.K_TRAIN,
                                      main_transform=None,
                                      img_transform=img_transform,
                                      gt_transform=gt_transform)
        val_loader = DataLoader(val_set,
                                batch_size=cfg_data.TEST_BS,
                                num_workers=cfg_data.N_WORKERS,
                                shuffle=False, drop_last=False)
        val_loaders.append(val_loader)

    test_loaders = []
    if test_adapt_imgs:
        for scene, adapt_imgs in zip(cfg_data.TEST_SCENES, test_adapt_imgs):
            test_set = WE_CSRNet_Meta_eval(cfg_data.DATA_PATH, 'test', scene, adapt_imgs=adapt_imgs,
                                           main_transform=None,
                                           img_transform=img_transform,
                                           gt_transform=gt_transform)
            test_loader = DataLoader(test_set,
                                     batch_size=cfg_data.TEST_BS,
                                     num_workers=cfg_data.N_WORKERS,
                                     shuffle=False, drop_last=False)
            test_loaders.append(test_loader)
    else:
        for scene in cfg_data.TEST_SCENES:
            test_set = WE_CSRNet_Meta_eval(cfg_data.DATA_PATH, 'test', scene, n_adapt_imgs=cfg_data.K_TRAIN,
                                           main_transform=None,
                                           img_transform=img_transform,
                                           gt_transform=gt_transform)
            test_loader = DataLoader(test_set,
                                     batch_size=cfg_data.TEST_BS,
                                     num_workers=cfg_data.N_WORKERS,
                                     shuffle=False, drop_last=False)
            test_loaders.append(test_loader)

    return train_loader, val_loaders, test_loaders, restore_transform
