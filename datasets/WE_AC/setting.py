from easydict import EasyDict as edict

# init
__C_WEAC = edict()

cfg_data = __C_WEAC

__C_WEAC.STD_SIZE = (576, 720)
__C_WEAC.TRAIN_SIZE = (512, 672)

__C_WEAC.DATA_PATH = 'D:\OneDrive\OneDrive - UvA\ThesisData\Datasets\WE_C3_PP_FSL'  # WorldExpo'10 preprocessed

__C_WEAC.MEAN_STD = ([0.504379212856, 0.510956227779, 0.505369007587], [0.22513884306, 0.225588873029, 0.22579960525])

__C_WEAC.LABEL_FACTOR = 1
__C_WEAC.LOG_PARA = 100.

__C_WEAC.N_UNLABELED = 1  # > 1 has not yet been implemented
__C_WEAC.N_LABELED = 5


