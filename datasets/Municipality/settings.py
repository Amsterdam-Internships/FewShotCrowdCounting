from easydict import EasyDict as edict

cfg_data = edict()

cfg_data.DATA_PATH = 'actual_data_datasets\\DAM_CrossVal_Combined-density_export'

cfg_data.TRAIN_BS = 50
cfg_data.TEST_BS = 1
cfg_data.N_WORKERS = 4

cfg_data.MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
cfg_data.LABEL_FACTOR = 10000

