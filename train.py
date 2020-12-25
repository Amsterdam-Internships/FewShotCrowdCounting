import os
import numpy as np
import torch

from config import cfg


def _get_dataloader(data_mode):
    loading_data, cfg_data = None, None  # Suppress warning
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
    elif data_mode == 'WE_AC':
        from datasets.WE_AC.loading_data import loading_data
        from datasets.WE_AC.setting import cfg_data
    else:
        print(f"data mode {data_mode} not recognised!")
        exit(1)

    return loading_data, cfg_data


def _get_trainer(training_mode):
    Trainer = None  # Suppress warning
    if training_mode == 'Trainer':
        from trainers.trainer import Trainer
    elif training_mode == 'WE_MAML_trainer':
        from trainers.trainer_for_MAML import Trainer
    elif training_mode == 'AC_trainer':
        from trainers.trainer_for_AdaCrowd import Trainer
    elif training_mode == 'SaNet_Trainer':
        from trainers.trainer_for_M2TCC import Trainer  # double losses but signle output
    elif training_mode == 'CMTL_Trainer':
        from trainers.trainer_for_CMTL import Trainer  # double losses and double outputs
    else:
        print(f'Trainer {training_mode} not recognised!')
        exit(1)

    return Trainer


def _get_network(model_name):
    net = None
    if model_name == 'AlexNet':
        from models.SCC_Model.AlexNet import AlexNet as net
    elif model_name == 'VGG':
        from models.SCC_Model.VGG import VGG as net
    elif model_name == 'VGG_DECODER':
        from models.SCC_Model.VGG_decoder import VGG_decoder as net
    elif model_name == 'MCNN':
        from models.SCC_Model.MCNN import MCNN as net
    elif model_name == 'CSRNet':
        from models.SCC_Model.CSRNet import CSRNet as net
    elif model_name == 'Res50':
        from models.SCC_Model.Res50 import Res50 as net
    elif model_name == 'Res101':
        from models.SCC_Model.Res101 import Res101 as net
    elif model_name == 'Res101_SFCN':
        from models.SCC_Model.Res101_SFCN import Res101_SFCN as net
    elif model_name == 'ACCSRNet':
        from models.SCC_Model.ACCSRNet import ACCSRNet as net
    else:
        print(f'model name {model_name} not recognised.')
        exit(1)

    return net


def _init_crowdcounter(cc_name, net, loss_func):
    crowdCounter = None
    if cc_name == 'CC':
        from models.CC import CrowdCounter
        crowdCounter = CrowdCounter()


    elif network == 'ACCSRNet':
        from models.SCC_Model.ACCSRNet import ACCSRNet as net
    elif network == 'MAML_CSRNet':
        from models import MAMLCC_Model as net  # Yeah, cheeky strategy, but it works
    elif network == 'DSNet':
        pass
    else:
        print(f'Network {network} not recognised!')
        exit(1)
    return CrowdCounter


def main():
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

    # ------------get training components-----------
    dataloader, cfg_data = _get_dataloader(cfg.DATASET)
    net = _get_network(cfg.MODEL_NAME)
    Trainer = _get_trainer(cfg.TRAINER)
    crowdCounter = _init_crowdcounter(cfg.CROWD_COUNTER, net, gpus)

    # ------------Start Training------------
    pwd = os.path.split(os.path.realpath(__file__))[0]
    cc_trainer = Trainer(dataloader, crowdCounter, cfg, cfg_data, pwd)
    cc_trainer.forward()


# Added because windows:
# https://stackoverflow.com/questions/18204782/runtimeerror-on-windows-trying-python-multiprocessing
if __name__ == '__main__':
    main()
