import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from models.CC import CrowdCounter
from misc.utils import *
import pdb
import time


class Trainer():
    def __init__(self, dataloader, network, CrowdCounter, cfg, cfg_data, pwd):
        self.cfg_data = cfg_data
        self.dataloader = dataloader

        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.MODEL_NAME
        network = network()
        print(network)
        self.cc_net = CrowdCounter(network, cfg.GPU_ID, cfg.LOSS_FUNCS, cfg=cfg)

        self.optimizer = optim.Adam(self.cc_net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        # self.optimizer = optim.SGD(self.net.parameters(), cfg.LR, momentum=0.95,weight_decay=5e-4)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)

        # self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}
        # self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}
        #
        # self.epoch = 0
        # self.i_tb = 0

        if cfg.RESUME:
            print("Resume not supported yet")
            exit(1)
            latest_state = torch.load(cfg.RESUME_PATH)
            self.cc_net.load_state_dict(latest_state['net'])
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.scheduler.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']

        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME)

    def forward(self):
        folds = [1, 2, 3, 4, 5]

        for validation_fold in folds:
            self.current_fold = validation_fold
            self.epoch = 0
            self.i_tb = 0
            self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}
            self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}

            self.cfg_data.VAL_INDEX = validation_fold
            self.train_loader, self.val_loader, self.restore_transform = self.dataloader()

            self.run_epochs()

            last_mae, last_mse = self.validate_ECF50()  # might result in redundant saves.
            save_name = f'fold_{validation_fold}_last_mae{last_mae:.1f}_mse_{last_mse:.1f}_.pth'
            torch.save(self.cc_net.state_dict(), os.path.join(self.exp_path, self.exp_name, save_name))



    def run_epochs(self):
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch

            # training
            self.timer['train time'].tic()
            self.train()
            self.timer['train time'].toc(average=False)

            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('=' * 20)

            # validation
            if epoch % cfg.VAL_FREQ == 0 or epoch > cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                self.validate_ECF50()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))

            if epoch > cfg.LR_DECAY_START:
                self.scheduler.step()

    def train(self):  # training for all datasets

        self.cc_net.train()
        for i, data in enumerate(self.train_loader, 0):
            self.timer['iter time'].tic()

            img, gt_map = data
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()

            self.optimizer.zero_grad()
            pred_map = self.cc_net(img, gt_map)
            loss = self.cc_net.loss
            loss.backward()
            self.optimizer.step()

            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                self.timer['iter time'].toc(average=False)
                print('[ep %d][it %d][loss %.4f][lr %.4f][%.2fs]' % \
                      (self.epoch + 1, i + 1, loss.item(), self.optimizer.param_groups[0]['lr'] * 10000,
                       self.timer['iter time'].diff))
                print('        [cnt: gt: %.1f pred: %.2f]' % (
                gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0].sum().data / self.cfg_data.LOG_PARA))

    def validate_ECF50(self):  # validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50

        self.cc_net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        for vi, data in enumerate(self.val_loader, 0):
            img, gt_map = data

            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()

                pred_map = self.cc_net.forward(img, gt_map)

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    losses.update(self.cc_net.loss.item())
                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))
                if vi == 0:
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)

        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.train_record = update_model(self.cc_net, self.optimizer, self.scheduler, self.epoch, self.i_tb,
                                         self.exp_path, self.exp_name,  [mae, mse, loss],
                                         self.train_record, self.log_txt, fold=self.current_fold)
        print_summary(self.exp_name, [mae, mse, loss], self.train_record)

        return mae, mse
