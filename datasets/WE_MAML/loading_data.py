import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import misc.transforms as own_transforms
from .WE_MAML import WEMAML
from .setting import cfg_data
from torchvision import transforms


def loading_data(task, mode):
    mean_std = cfg_data.MEAN_STD
    log_para = cfg_data.LOG_PARA

    img_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    gt_transform = standard_transforms.Compose([
        own_transforms.LabelNormalize(log_para)
    ])

    if mode == 'train':
        imgs = task.train_images
    else:
        imgs = task.validation_images

    train_set = WEMAML(imgs, mode, img_transform=img_transform, gt_transform=gt_transform)
    dataloader = DataLoader(train_set, batch_size=1, num_workers=8, shuffle=True, drop_last=True)
    # TODO: batch size = 12 and dynamic

    return dataloader


# train_set = WEMAML(task.train_images, 'train', main_transform=train_main_transform)
# test_name = cfg_data.VAL_FOLDER
#
# val_loader = []
#
# for subname in test_name:
#     sub_set = WE(cfg_data.DATA_PATH + '/test/' + subname, 'test', main_transform=val_main_transform,
#                  img_transform=img_transform, gt_transform=gt_transform)
#     val_loader.append(
#         DataLoader(sub_set, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=8, shuffle=True, drop_last=True))
