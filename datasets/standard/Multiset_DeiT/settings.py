from easydict import EasyDict as edict

cfg_data = edict()

cfg_data.TRAIN_BASE_PATHS = [
    'D:\\ThesisData\\Datasets\\Arena 02',
    'D:\\ThesisData\\Datasets\\Arena 03',
    'D:\\ThesisData\\Datasets\\Arena 03',
    'D:\\ThesisData\\Datasets\\Arena 03'
]

cfg_data.VAL_BASE_PATHS = [
    'D:\\ThesisData\\Datasets\\DAM_competition'
]

cfg_data.TEST_BASE_PATHS = [
    'D:\\ThesisData\\Datasets\\CV Arena 05'
]

cfg_data.TRAIN_BS = 10
cfg_data.VAL_BS = 1
cfg_data.N_WORKERS = 0

cfg_data.MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
cfg_data.LABEL_FACTOR = 3000

cfg_data.OVERLAP = 8  # For test images, how much overlap should crops have
cfg_data.IGNORE_BUFFER = 4  # When reconstructing the complete density map, how many pixels of edges should be ignored
                             # No pixels are ignored at the edges of the density map.

