from easydict import EasyDict as edict

# init
__C_WEMAML = edict()

cfg_data = __C_WEMAML

__C_WEMAML.STD_SIZE = (576, 720)
__C_WEMAML.TRAIN_SIZE = (512, 672)

__C_WEMAML.DATA_PATH = 'D:\OneDrive\OneDrive - UvA\ThesisData\Datasets\WE_C3_PP_FSL'  # WorldExpo'10 preprocessed

# From original WE, on which the model's weights are initialised
__C_WEMAML.MEAN_STD = ([0.504379212856, 0.510956227779, 0.505369007587], [0.22513884306, 0.225588873029, 0.22579960525])

# __C_WEMAML.LABEL_FACTOR = 1
__C_WEMAML.LOG_PARA = 10000.

__C_WEMAML.VAL_FOLDER = ['104207', '200608', '200702', '202201', '500717']

__C_WEMAML.NUM_OF_INSTANCES = 5
__C_WEMAML.NUM_OF_TASKS = 8

__C_WEMAML.META_BATCHSIZE = 5
__C_WEMAML.BASE_BATCHSIZE = 1
