import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

from datasets.dataset_utils import img_equal_unsplit
import matplotlib.pyplot as plt
from matplotlib import cm as CM


class Trainer:
    def __init__(self, model, model_funct, loading_data, cfg, cfg_data):
        self.model = model
        self.model_funct = model_funct
        self.cfg = cfg
        self.cfg_data = cfg_data

        self.train_loader, self.test_loader, self.restore_transform = loading_data(self.model.crop_size)
        self.train_samples = len(self.train_loader.dataset)
        # self.test_samples = len(self.test_loader.dataset) # TODO
        # self.eval_save_example_every = self.test_samples // self.cfg.SAVE_NUM_EVAL_EXAMPLES

        self.alpha = self.cfg.ALPHA
        self.beta = self.cfg.BETA  # TODO: MAKE IN CONFIG FILE
        self.criterion = torch.nn.MSELoss()
        self.meta_optimiser = torch.optim.Adam(model.parameters(), lr=self.beta, weight_decay=cfg.WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.meta_optimiser, step_size=1, gamma=cfg.LR_GAMMA)

        self.n_tasks = cfg.N_TASKS

        self.epoch = 0
        self.best_mae = 10 ** 10  # just something high
        self.best_epoch = -1

        self.writer = SummaryWriter(cfg.SAVE_DIR)

        if cfg.RESUME:
            self.load_state(cfg.RESUME_PATH)
            print(f'Resuming from epoch {self.epoch}')
        # else:
        #     self.save_eval_pics()
        #     self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], self.epoch)

        # self.test_functional()

    def test_functional(self):
        """ Just a small function so I can verify with Pycharm debugger if functional model even works"""

        tasks_sampler = iter(self.train_loader)
        img, gt, _, _ = next(tasks_sampler)
        img = img.squeeze()
        model_weights = self.model.state_dict()
        # model_pred = self.model.forward(img)
        model_func_pred = self.model_funct.forward(img, model_weights, True)
        print('done')

    def inner_loop(self, train_img, train_gt, test_img, test_gt, theta, theta_weights):
        train_img, train_gt = train_img.squeeze().cuda(), train_gt.squeeze().cuda()
        test_img, test_gt = test_img.squeeze().cuda(), test_gt.squeeze().cuda()

        train_pred = self.model_funct.forward(train_img, theta, True)  # training is True
        train_pred = train_pred.squeeze(1)  # Remove channel dim
        train_loss = self.criterion(train_pred, train_gt)
        grads = torch.autograd.grad(train_loss, theta_weights)

        theta_prime = OrderedDict((k, w - self.alpha * g) for k, w, g in zip(theta.keys(), theta.values(), grads))

        test_pred = self.model_funct.forward(test_img, theta_prime, True)
        test_pred = test_pred.squeeze(1)
        test_loss = self.criterion(test_pred, test_gt)

        AEs_ = (torch.sum(test_pred.detach(), dim=(1, 2)) - torch.sum(test_gt, dim=(1, 2))) / self.cfg_data.LABEL_FACTOR
        AEs = torch.mean(torch.abs(AEs_)).cpu().item()

        return test_loss, AEs

    def train(self):  # Outer loop
        items_left = 0
        tasks_sampler = None

        self.model.train()
        while self.epoch < self.cfg.MAX_EPOCH:  # While not done
            self.epoch += 1
            theta = OrderedDict((name, param) for name, param in self.model.named_parameters())
            del theta['norm.weight']
            del theta['norm.bias']
            del theta['head.weight']
            del theta['head.bias']
            theta_weights = list(theta.values())

            # Check if enough tasks available
            if items_left < self.n_tasks:
                tasks_sampler = iter(self.train_loader)
                items_left = len(self.train_loader.dataset)

            total_metaloss = torch.tensor(0).float().cuda()
            total_AEs = 0
            for task_idx in range(self.n_tasks):
                train_img, train_gt, test_img, test_gt = next(tasks_sampler)
                items_left -= 1

                test_loss, AEs = self.inner_loop(train_img, train_gt, test_img, test_gt, theta, theta_weights)
                total_metaloss += test_loss
                total_AEs += AEs

            avg_metaloss = total_metaloss / self.n_tasks
            metagrads = torch.autograd.grad(avg_metaloss, theta_weights)

            for w, g in zip(theta_weights, metagrads):
                w.grad = g
            self.meta_optimiser.step()

            print(f'ep {self.epoch}: Mean Patch Error: {total_AEs / self.n_tasks:.3f}')

            if self.epoch % self.cfg.SAVE_EVERY == 0:
                self.save_state()
        self.save_state()

    def save_state(self, name_extra=''):
        if name_extra:
            save_name = f'{self.cfg.STATE_DICTS_DIR}/save_state_ep_{self.epoch}_{name_extra}.pth'
        else:
            save_name = f'{self.cfg.STATE_DICTS_DIR}/save_state_ep_{self.epoch}.pth'

        save_sate = {
            'epoch': self.epoch,
            'best_epoch': self.best_epoch,
            'best_mae': self.best_mae,
            'net': self.model.state_dict(),
            'optim': self.meta_optimiser.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'save_dir_path': self.cfg.SAVE_DIR,
        }

        torch.save(save_sate, save_name)

    def load_state(self, state_path):
        resume_state = torch.load(state_path)
        self.epoch = resume_state['epoch']
        self.best_epoch = resume_state['best_epoch']
        self.best_mae = resume_state['best_mae']

        self.model.load_state_dict(resume_state['net'])
        self.meta_optimiser.load_state_dict(resume_state['optim'])
        self.scheduler.load_state_dict(resume_state['scheduler'])


def print_fancy_new_best_MAE():
    """ For that extra bit of dopamine rush when you get a new high-score"""

    new_best = '#' + '=' * 20 + '<' * 3 + ' NEW BEST MAE ' + '>' * 3 + '=' * 20 + '#'
    n_chars = len(new_best)
    bar = '#' + '=' * (n_chars - 2) + '#'
    print(bar)
    print(new_best)
    print(bar)
