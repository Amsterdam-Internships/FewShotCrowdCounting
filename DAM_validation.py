import cv2, platform
import numpy as np
import urllib
import torchvision.transforms.functional as F
import os
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import torch
import IPython.display as display
from matplotlib import pyplot as plt
from matplotlib import cm as CM
import gc
import time
import csv

import datasets.transforms as own_transforms
import torchvision.transforms as standard_transforms
from datasets.dataset_utils import split_image_and_den, unsplit_den
from timm.models import create_model
import custom_models

MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
LABEL_FACTOR = 10000
PATCH_SIZE = 224  # 224 or 384

img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*MEAN_STD)
])

restore_transform = standard_transforms.Compose([
    own_transforms.DeNormalize(*MEAN_STD),
    standard_transforms.ToPILImage()
])


def load_frames():
    root = 'actual_data_datasets/DAM_validation'
    subs = [
        'PC5-1731-GADM-01-Dam_20200918_213000818-MASKED',
        'PC5-1731-GADM-01-Dam_20200919_120000836-MASKED',
        'PC5-1731-GADM-01-Dam_20200919_124500855-MASKED',
        'PC5-1731-GADM-01-Dam_20200919_173000949-MASKED',
        'PC5-1731-GADM-01-Dam_20200920_143001120-MASKED',
        'RAIN_PC5-1731-GADM-01-Dam_20200923_174500350-MASKED'
    ]
    paths = []
    for sub in subs:
        full_path = os.path.join(root, sub)
        for frame in os.listdir(full_path):
            paths.append(os.path.join(full_path, frame))
    paths.sort()
    return paths


def get_model():
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

    return model


def read_image(img_path):
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')

    img = img.crop((750, 1100, 750 + 5600, 1100 + 2740))

    new_w, new_h = 1866, 874

    return img.resize((new_w, new_h))


def process_getitem_test(img):
    w, h = img.size
    fake_den = np.zeros((h, w))
    fake_den = Image.fromarray(fake_den)
    imgs, dens = split_image_and_den(img, fake_den, 224)
    for i in range(len(imgs)):
        if img_transform:  # These should always be provided
            imgs[i] = img_transform(imgs[i])
    return torch.stack(imgs), torch.tensor(img.size)


def predict_from_image2(img_path, network):
    img = read_image(img_path)
    img_stack, img_resolution = process_getitem_test(img)

    img_stack = img_stack.cuda()
    output = network(img_stack)
    output = output.detach().squeeze().cpu()
    predicted_den, gt = unsplit_den(output, output, img_resolution)

    output[output < 0] = 0
    width, height = img.size

    count = predicted_den.sum() / LABEL_FACTOR

    output = cv2.resize(np.float32(output), (width, height))

    return count, output, img

def test(model, path):
    count, densMap, image = predict_from_image2(path, model)

    #return count, image, imageWithText, densMap
    return count, image, densMap





def main():
    errs = []
    gts = []
    data = load_frames()
    model = get_model()

    model.eval()
    with torch.no_grad():
        with open('dam-validation-roi-2-FINAL.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['image', 'prediction'])
            for d in data[:]:

                # count, image, imgWithText, densMap = test(WEIGHTS, d, False, None)
                count, image, densMap = test(model, d)

                # display_fig(image, densMap)#, densMap)

                if count < 0:
                    count = 0.0

                print(d, 'prediction:', count)

                imgname = d.replace("/home/eagle/dam_scaled/", "")

                writer.writerow([imgname, count])

                csvfile.flush()

def my_test():
    img = Image.open('actual_data_datasets/DAM_validation/PC5-1731-GADM-01-Dam_20200918_213000818-MASKED/img_00001.jpg')
    if img.mode == 'L':
        img = img.convert('RGB')

    img = img.crop((750, 1100, 750 + 5600, 1100 + 2740))

    new_w, new_h = 1866, 930

    return img.resize((new_w, new_h))

if __name__ == '__main__':
    # main()
    im = my_test()
    im.show()