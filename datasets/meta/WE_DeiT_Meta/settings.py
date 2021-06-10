from easydict import EasyDict as edict
import os

cfg_data = edict()

cfg_data.DATA_PATH = 'D:\\OneDrive\\OneDrive - UvA\\ThesisData\\Datasets\\WE_C3_Meta'

cfg_data.VAL_SCENES = os.listdir(os.path.join(cfg_data.DATA_PATH, 'val'))  # All validation scenes
cfg_data.TEST_SCENES = os.listdir(os.path.join(cfg_data.DATA_PATH, 'test'))  # All test scenes

cfg_data.TRAIN_BS = 1  # must be 1
cfg_data.TEST_BS = 1  # must be 1
cfg_data.N_WORKERS = 4

cfg_data.K_TRAIN = 0  # D
cfg_data.K_META = 2   # D'

cfg_data.MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
cfg_data.LABEL_FACTOR = 3000

cfg_data.CROP_SIZE = 224
cfg_data.OVERLAP = 8        # For test images, how much overlap should crops have
cfg_data.IGNORE_BUFFER = 4  # When reconstructing the complete density map, how many pixels of edges should be ignored
# # No pixels are ignored at the edges of the density map.
