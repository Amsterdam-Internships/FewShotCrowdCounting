import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from models.SCC_Model.ACCSRNet import ACCSRNet
from config import cfg
from misc.utils import *
import pdb
import time


class Trainer():
    def __init__(self, dataloader, cfg_data, pwd):

        self.cfg_data = cfg_data

        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.NET
        self.my_net = ACCSRNet().cuda()

        # self.optimizer = optim.Adam(self.my_net.parameters(), lr=cfg.LR, weight_decay=1e-4)
        self.optimizer = optim.Adam(self.my_net.parameters(), lr=cfg.LR)  # Paper does not mention weight decay
        # self.optimizer = optim.SGD(self.net.parameters(), cfg.LR, momentum=0.95,weight_decay=5e-4)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)
        self.criterion = nn.MSELoss()

        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}
        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}

        self.epoch = 0
        self.i_tb = 0

        if cfg.PRE_GCC:
            self.net.load_state_dict(torch.load(cfg.PRE_GCC_MODEL))

        self.train_loader, self.val_loader, self.restore_transform = dataloader()

        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH)
            self.net.load_state_dict(latest_state['net'])
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.scheduler.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']

        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME)

    def forward(self):

        self.validate_V4()
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch
            if epoch > cfg.LR_DECAY_START:
                self.scheduler.step()

            # training
            self.timer['train time'].tic()
            self.train()
            self.timer['train time'].toc(average=False)

            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('=' * 20)

            # validation
            if epoch % cfg.VAL_FREQ == 0 or epoch > cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                self.validate_V4()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))

    def train(self):  # training for all datasets
        self.my_net.train()
        for i, data in enumerate(self.train_loader, 0):
            self.timer['iter time'].tic()

            unl_img, img, gt_map = data
            unl_img = torch.cat(unl_img).cuda()
            img = torch.cat(img).cuda()
            gt_map = torch.cat(gt_map).cuda()

            self.optimizer.zero_grad()
            pred_map = self.my_net.forward(unl_img, img)
            loss = self.criterion(pred_map, gt_map)
            loss.backward()
            nn.utils.clip_grad_norm_(self.my_net.parameters(), 1.)  # Clip grad norm to 1, like in the paper.
            self.optimizer.step()

            print(f'pred: {pred_map.sum() / 100}, gt: {gt_map.sum() / 100}')
            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                self.timer['iter time'].toc(average=False)
                print('[ep %d][it %d][loss %.4f][lr %.4f][%.2fs]' % \
                      (self.epoch + 1, i + 1, loss.item(), self.optimizer.param_groups[0]['lr'] * 10000,
                       self.timer['iter time'].diff))
                print('        [cnt: gt: %.1f pred: %.2f]' % (
                gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0].sum().data / self.cfg_data.LOG_PARA))

    # TODO: Evaluation is nowhere near optimal. It just does the same as training but with test set. NOT GOOD!!!
    def validate_V4(self):  # validate_V2 for WE

        self.my_net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        # ROI mask is not used!
        # roi_mask = []
        from datasets.WE_AC.setting import cfg_data
        # from scipy import io as sio
        # for val_folder in cfg_data.VAL_FOLDER:
        #     roi_mask.append(sio.loadmat(os.path.join(cfg_data.DATA_PATH, 'test', val_folder + '_roi.mat'))['BW'])
        #
        #
        # mask = roi_mask[i_sub]
        for vi, data in enumerate(self.val_loader, 0):
            unl_img, img, gt_map = data
            print(vi)
            with torch.no_grad():
                unl_img = torch.cat(unl_img).cuda()
                img = torch.cat(img).cuda()
                gt_map = torch.cat(gt_map)

                pred_map = self.my_net.forward(unl_img, img)

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    # losses.update(self.my_net.loss.item())
                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))
                if vi == 0:
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)

        mae = maes.avg
        mse = np.sqrt(mses.avg)
        # loss = losses.avg

        # self.writer.add_scalar('val_loss', loss, self.epoch + 1)  # Not added to ACCSRNet !
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.train_record = update_model(self.my_net, self.optimizer, self.scheduler, self.epoch, self.i_tb,
                                         self.exp_path, self.exp_name,
                                         [mae, mse, -1], self.train_record, self.log_txt)
        print_summary(self.exp_name, [mae, mse, -1], self.train_record)
        print("done")