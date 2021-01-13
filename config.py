import os
from easydict import EasyDict as edict
import time
import torch

# init
__C = edict()
cfg = __C

# ========================================================================================== #
# ===================================      TRAIN     ======================================= #
# ========================================================================================== #

__C.SEED = 42  # random seed,  for reproduction

# DATASET INFO
__C.DATASET = 'WE_MAML'

# Model info
__C.MODEL_NAME = 'CSRNet_MAML'

# Trainer info
__C.TRAINER = 'WE_MAML_trainer'

# Which crowd counter to use (models.XYZ)
__C.CROWD_COUNTER = 'CC'

# Loss functions can be more than one depending on which crowdcounter you use
__C.LOSS_FUNCS = ['MSELoss']

# learning rate settings
# __C.LR = 5e-6

__C.META_LR = 0.001
__C.BASE_LR = 0.001
__C.LR = '0.001'  # For the logger / prints

__C.LR_DECAY = 1  # decay rate
__C.LR_DECAY_START = -1  # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 10**10  # decay frequency
__C.MAX_EPOCH = 1500


# print
__C.PRINT_FREQ = 100

now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_PATH = './exp'
__C.EXP_NAME = now \
               + '_' + __C.DATASET \
               + '_' + __C.MODEL_NAME \
               + '_' + str(__C.LR)


# ========================================================================================== #
# ===================================      EVAL      ======================================= #
# ========================================================================================== #
__C.VAL_DENSE_START = 50
__C.VAL_FREQ = 5  # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ

# ------------------------------VIS------------------------
__C.VISIBLE_NUM_IMGS = 1  # must be 1 for training images with the different sizes

__C.RESUME = True
# __C.RESUME_PATH = "./exp/12-10_15-16_WE_CSRNet_1e-05/all_ep_16_mae_14.2_mse_0.0.pth"
__C.RESUME_PATH = 'CSRNet_SGD_STANDARD_100_epochs.pth'


# ========================================================================================== #
# ===================================      Other      ====================================== #
# ========================================================================================== #
__C.GPU_ID = [0]


__C.NUM_TASKS = 8
__C.BASE_UPDATES = 10

