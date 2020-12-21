import os
from easydict import EasyDict as edict
import time
import torch

# init
__C = edict()
cfg = __C

# ------------------------------TRAIN------------------------
__C.SEED = 42  # random seed,  for reporduction
__C.DATASET = 'WE_AC'  # dataset selection: GCC, SHHA, SHHB, UCF50, UCFQNRF, WE, WE_MAML, WE_AC

if __C.DATASET == 'UCF50':  # only for UCF50
    from datasets.UCF50.setting import cfg_data

    __C.VAL_INDEX = cfg_data.VAL_INDEX

if __C.DATASET == 'GCC':  # only for GCC
    from datasets.GCC.setting import cfg_data

    __C.VAL_MODE = cfg_data.VAL_MODE

__C.NET = 'ACCSRNet'  # net selection: MCNN, VGG, VGG_DECODER, CSRNet, ACCSRNet

__C.PRE_GCC = False  # use the pretrained model on GCC dataset
__C.PRE_GCC_MODEL = ''  # path to model

__C.GPU_ID = [0]  # sigle gpu: [0], [1] ...; multi gpus: [0,1]

# learning rate settings
__C.META_LR = 0.001  # META_LR
__C.BASE_LR = 0.001  # BASE_LR
__C.LR_DECAY = 1  # decay rate
__C.LR_DECAY_START = -1  # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 10**10  # decay frequency
__C.MAX_EPOCH = 3000
# __C.LR = str(__C.META_LR) + str(__C.BASE_LR)
__C.LR = 1e-5

# Batch settings
# __C.TEST_BATCH_SIZE = 1  # imgs
# __C.META_BATCH = 5  # META_BATCHSIZE
# __C.BASE_BATCH = 1  # BASE_BATCHSIZE


# print
__C.PRINT_FREQ = 10

now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = now \
               + '_' + __C.DATASET \
               + '_' + __C.NET \
               + '_' + str(__C.LR)

if __C.DATASET == 'UCF50':
    __C.EXP_NAME += '_' + str(__C.VAL_INDEX)

if __C.DATASET == 'GCC':
    __C.EXP_NAME += '_' + __C.VAL_MODE

__C.EXP_PATH = './exp'  # the path of logs, checkpoints, and current codes

# ------------------------------VAL------------------------
__C.VAL_DENSE_START = 50
__C.VAL_FREQ = 5  # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ

# ------------------------------VIS------------------------
__C.VISIBLE_NUM_IMGS = 1  # must be 1 for training images with the different sizes

# __C.RESUME = True
__C.RESUME = False
# __C.RESUME_PATH = "./exp/12-10_15-16_WE_CSRNet_1e-05/all_ep_16_mae_14.2_mse_0.0.pth"

__C.EPOCHS = 15  # Number of epochs
__C.BASE_UPDATES = 10  # BASE_UPDATES
__C.DATASET_FULL_NAME = "WE_C3_PP_FSL"

__C.TRAINING_MODE = ''
# ================================================================================
# ================================================================================
# ================================================================================
