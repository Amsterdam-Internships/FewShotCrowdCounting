import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import misc.transforms as own_transforms
from .WE_MAML import WEMAML
from .setting import cfg_data
from torchvision import transforms
import os

def loading_data(task, mode='train'):
    mean_std = cfg_data.MEAN_STD
    log_para = cfg_data.LOG_PARA
    batch_size = cfg_data.TRAIN_BATCH_SIZE

    img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])

    gt_transform = standard_transforms.Compose([own_transforms.LabelNormalize(log_para)])

    if mode == 'train':
        imgs = task.train_images
    else:
        imgs = task.validation_images

    dataset = WEMAML(imgs, mode, img_transform=img_transform, gt_transform=gt_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)

    return dataloader


def retrieve_test_images():
    frames_path = os.path.join(cfg_data.DATA_PATH, 'test', 'frames')
    folders = [os.path.join(frames_path, frame) for frame in os.listdir(frames_path)]
    images = []
    for folder in folders:
        imgs = [os.path.join(folder, img) for img in os.listdir(folder)]
        images += imgs

    return images


def loading_test_data():
    mean_std = cfg_data.MEAN_STD
    log_para = cfg_data.LOG_PARA
    batch_size = cfg_data.TEST_BATCH_SIZE

    img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])
    gt_transform = standard_transforms.Compose([own_transforms.LabelNormalize(log_para)])

    images = retrieve_test_images()

    dataset = WEMAML(images, 'test', img_transform=img_transform, gt_transform=gt_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8)

    return dataloader
