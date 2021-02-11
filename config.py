from easydict import EasyDict as edict
import time
import os

cfg = edict()



# __all__ = [
#     'deit_base_patch16_224',              'deit_small_patch16_224',               'deit_tiny_patch16_224',
#     'deit_base_distilled_patch16_224',    'deit_small_distilled_patch16_224',     'deit_tiny_distilled_patch16_224',
#     'deit_base_patch16_384',                                                      'deit_tiny_cnn_patch16_224',
#     'deit_base_distilled_patch16_384',
# ]

cfg.SEED = 42

cfg.MODEL = 'deit_tiny_patch16_224'
cfg.MODEL_FUNCTIONAL = 'deit_tiny_patch16_224_functional'
cfg.DATASET = 'WE_MAML'

cfg.COUNT_LOSS_FACTOR = 0
cfg.ALPHA = 0.01  # Inner update
cfg.BETA = 0.001  # Outer/meta update
cfg.LR_GAMMA = 0.1  # Scale LR by this at each step epoch
cfg.LR_STEP_EPOCHS = [2000, 7500, 20000]
cfg.WEIGHT_DECAY = 1e-5

cfg.N_TASKS = 5  # How many tasks to perform before performing an outer backprop


cfg.MAX_EPOCH = 25000
cfg.EVAL_EVERY = 1000  # Not yet implemented, but used still for save every!
cfg.SAVE_EVERY_N_EVALS = 3  # Every Nth evaluation, save model regardless of performance
cfg.SAVE_EVERY = cfg.SAVE_EVERY_N_EVALS * cfg.EVAL_EVERY  # Don't touch this one

cfg.SAVE_NUM_EVAL_EXAMPLES = 10  # How many examples from the test/evaluation set to save

# ===================================================================================== #
#                                 SAVE DIRECTORIES                                      #
# ===================================================================================== #
runs_dir = 'runs'
if not os.path.exists(runs_dir):
    os.mkdir(runs_dir)
    with open(os.path.join(runs_dir, '__init__.py'), 'w') as f:  # For dynamic loading of config file
        pass

cfg.SAVE_DIR = os.path.join(runs_dir, time.strftime("%m-%d_%H-%M", time.localtime()))
cfg.PICS_DIR = os.path.join(cfg.SAVE_DIR, 'pics')
cfg.STATE_DICTS_DIR = os.path.join(cfg.SAVE_DIR, 'state_dicts')
cfg.CODE_DIR = os.path.join(cfg.SAVE_DIR, 'code')


cfg.RESUME = False
cfg.RESUME_DIR = os.path.join('runs', '02-03_18-43')
cfg.RESUME_STATE = 'save_state_ep_200_new_best_MAE_2.002.pth'
cfg.RESUME_PATH = os.path.join('runs', cfg.RESUME_DIR, 'state_dicts', cfg.RESUME_STATE)



