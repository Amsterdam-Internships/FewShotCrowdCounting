from easydict import EasyDict as edict

# init
__C_WE = edict()

cfg_data = __C_WE

__C_WE.STD_SIZE = (576, 720)
__C_WE.TRAIN_SIZE = (512, 672)
# __C_WE.DATA_PATH = 'D:\OneDrive\OneDrive - UvA\ThesisData\Datasets'  # WorldExpo'10 preprocessed
__C_WE.DATA_PATH = 'D:\OneDrive\OneDrive - UvA\ThesisData\Datasets\WE_C3_PP'  # WorldExpo'10 preprocessed

__C_WE.MEAN_STD = ([0.504379212856, 0.510956227779, 0.505369007587], [0.22513884306, 0.225588873029, 0.22579960525])

__C_WE.LABEL_FACTOR = 1
__C_WE.LOG_PARA = 10000.

__C_WE.RESUME_MODEL = ''  # model path
__C_WE.TRAIN_BATCH_SIZE = 1  # imgs

__C_WE.VAL_BATCH_SIZE = 1  #

__C_WE.VAL_FOLDER = ['104207', '200608', '200702', '202201', '500717']

# __C_WE.NUM_OF_INSTANCES = 5
# __C_WE.NUM_OF_TASKS = 10
