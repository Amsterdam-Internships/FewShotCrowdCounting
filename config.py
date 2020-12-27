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
__C.DATASET = 'SHHB2'

# Model info
__C.MODEL_NAME = 'DSNet'

# Trainer info
__C.TRAINER = 'Trainer'

# Which crowd counter to use (models.XYZ)
__C.CROWD_COUNTER = 'CC'

# Loss functions can be more than one depending on which crowdcounter you use
__C.LOSS_FUNCS = ['MSELoss']

# learning rate settings
__C.LR = 1e-5

__C.LR_DECAY = 1  # decay rate
__C.LR_DECAY_START = -1  # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 10**10  # decay frequency
__C.MAX_EPOCH = 3000


# print
__C.PRINT_FREQ = 10

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

__C.RESUME = False
# __C.RESUME_PATH = "./exp/12-10_15-16_WE_CSRNet_1e-05/all_ep_16_mae_14.2_mse_0.0.pth"


# ========================================================================================== #
# ===================================      Other      ====================================== #
# ========================================================================================== #
__C.GPU_ID = [0]
