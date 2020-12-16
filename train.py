import os
import numpy as np
import torch

from config import cfg

if __name__ == '__main__':  # TODO: Added because windows: https://stackoverflow.com/questions/18204782/runtimeerror-on-windows-trying-python-multiprocessing
    # ------------prepare enviroment------------
    seed = cfg.SEED
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    gpus = cfg.GPU_ID
    if len(gpus) == 1:
        torch.cuda.set_device(gpus[0])

    torch.backends.cudnn.benchmark = True

    # ------------prepare data loader------------
    data_mode = cfg.DATASET
    if data_mode == 'SHHA':
        from datasets.SHHA.loading_data import loading_data
        from datasets.SHHA.setting import cfg_data
    elif data_mode == 'SHHB':
        from datasets.SHHB.loading_data import loading_data
        from datasets.SHHB.setting import cfg_data
    elif data_mode == 'QNRF':
        from datasets.QNRF.loading_data import loading_data
        from datasets.QNRF.setting import cfg_data
    elif data_mode == 'UCF50':
        from datasets.UCF50.loading_data import loading_data
        from datasets.UCF50.setting import cfg_data
    elif data_mode == 'WE':
        from datasets.WE.loading_data import loading_data
        from datasets.WE.setting import cfg_data
    elif data_mode == 'GCC':
        from datasets.GCC.loading_data import loading_data
        from datasets.GCC.setting import cfg_data
    elif data_mode == 'Mall':
        from datasets.Mall.loading_data import loading_data
        from datasets.Mall.setting import cfg_data
    elif data_mode == 'UCSD':
        from datasets.UCSD.loading_data import loading_data
        from datasets.UCSD.setting import cfg_data
    elif data_mode == 'WE_MAML':
        from datasets.WE_MAML.loading_data import loading_data
        from datasets.WE_MAML.setting import cfg_data

    # ------------Prepare Trainer------------
    net = cfg.NET
    training_mode = cfg.TRAINING_MODE
    if data_mode == 'WE_MAML':
        from trainer_for_MAML import Trainer
    else:
        if net in ['MCNN', 'AlexNet', 'VGG', 'VGG_DECODER', 'Res50', 'Res101', 'CSRNet', 'Res101_SFCN']:
            from trainer import Trainer
        elif net in ['SANet']:
            from trainer_for_M2TCC import Trainer  # double losses but signle output
        elif net in ['CMTL']:
            from trainer_for_CMTL import Trainer  # double losses and double outputs
        elif net in ['PCCNet']:
            from trainer_for_M3T3OCC import Trainer

    # ------------Start Training------------
    pwd = os.path.split(os.path.realpath(__file__))[0]
    cc_trainer = Trainer(loading_data, cfg_data, pwd)
    cc_trainer.forward()
