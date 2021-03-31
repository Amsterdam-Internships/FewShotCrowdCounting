from easydict import EasyDict as edict
import os

cfg_data = edict()

cfg_data.TRAIN_BS = 1
cfg_data.TEST_BS = 1
cfg_data.N_WORKERS = 0

cfg_data.K_TRAIN = 10  # D
cfg_data.K_META = 10   # D'

cfg_data.LABEL_FACTOR = 1

