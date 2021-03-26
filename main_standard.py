# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os

import models.DeiT.DeiTModels  # Needed to register models for 'create_model'
from timm.models import create_model

import importlib

from trainer_standard import Trainer
from config import cfg
from shutil import copyfile
import random

# __all__ = [
#     'deit_base_patch16_224',              'deit_small_patch16_224',               'deit_tiny_patch16_224',
#     'deit_base_distilled_patch16_224',    'deit_small_distilled_patch16_224',     'deit_tiny_distilled_patch16_224'
#     'deit_base_patch16_384',
#     'deit_base_distilled_patch16_384',
# ]


model_mapping = {
    'deit_tiny_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
    'deit_tiny_distilled_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
    'deit_small_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
    'deit_small_distilled_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
    'deit_base_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
    'deit_base_distilled_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
    'deit_base_patch16_384': 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
    'deit_base_distilled_patch16_384': 'https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth'
}


def make_save_dirs(loaded_cfg):
    """ Each run has its own directory structure, which is created here."""

    if not os.path.exists(loaded_cfg.SAVE_DIR):
        os.mkdir(loaded_cfg.SAVE_DIR)
        os.mkdir(loaded_cfg.PICS_DIR)
        os.mkdir(loaded_cfg.STATE_DICTS_DIR)
        os.mkdir(loaded_cfg.CODE_DIR)
        with open(os.path.join(cfg.SAVE_DIR, '__init__.py'), 'w') as f:  # For dynamic loading of config file
            pass

    else:
        print('save directory already exists!')


def main(cfg):
    """ Loads the settings and model, then creates a trainer with which the model is trained."""

    if cfg.RESUME:  # Not fully tested yet
        module = importlib.import_module(cfg.RESUME_DIR.replace(os.sep, '.') + 'code.config')
        cfg = module.cfg
    else:  # Make a backup of some important files for archiving purposes.
        make_save_dirs(cfg)
        copyfile('config.py', os.path.join(cfg.CODE_DIR, 'config.py'))
        copyfile('trainer_standard.py', os.path.join(cfg.CODE_DIR, 'trainer_standard.py'))
        copyfile('models/DeiT/DeiTModels.py', os.path.join(cfg.CODE_DIR, 'DeiTModels.py'))
        copyfile(os.path.join('datasets', 'standard', cfg.DATASET, 'settings.py'),
                 os.path.join(cfg.CODE_DIR, 'settings.py'))
        copyfile(os.path.join('datasets', 'standard', cfg.DATASET, 'loading_data.py'),
                 os.path.join(cfg.CODE_DIR, 'loading_data.py'))
        copyfile(os.path.join('datasets', 'standard', cfg.DATASET, cfg.DATASET + '.py'),
                 os.path.join(cfg.CODE_DIR, cfg.DATASET + '.py'))

    # fix the seed for reproducibility
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    cudnn.benchmark = True  # For efficiency

    print(f"Creating model: {cfg.MODEL}")

    # Default settings from the original DeiT framework
    model = create_model(
        cfg.MODEL,
        init_path=model_mapping[cfg.MODEL],
        num_classes=1000,  # Not yet used anyway. But must match pretrained model!
        drop_rate=0.,
        drop_path_rate=0.1,
        drop_block_rate=None,
    )

    model.cuda()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # Dynamically load the dataloader and its settings as specified in the config file
    dataloader = importlib.import_module(f'datasets.standard.{cfg.DATASET}.loading_data').loading_data
    cfg_data = importlib.import_module(f'datasets.standard.{cfg.DATASET}.settings').cfg_data

    trainer = Trainer(model, dataloader, cfg, cfg_data)  # Make a trainer
    trainer.train()  # Train the model


if __name__ == '__main__':
    main(cfg)
