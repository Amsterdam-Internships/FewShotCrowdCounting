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
    def __init__(self, meta_wrapper, loading_data, cfg, cfg_data):
        self.meta_wrapper = meta_wrapper

        self.cfg = cfg
        self.cfg_data = cfg_data

        self.train_loader, self.val_loaders, self.test_loaders, self.restore_transform = loading_data()
        self.train_samples = len(self.train_loader.dataset)
        # self.eval_save_example_every = self.test_samples // self.cfg.SAVE_NUM_EVAL_EXAMPLES

        self.beta = self.cfg.BETA
        self.meta_optimiser = torch.optim.Adam(self.meta_wrapper.get_params(),
                                               lr=self.beta, weight_decay=cfg.WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.meta_optimiser, step_size=1, gamma=cfg.LR_GAMMA)

        self.n_tasks = cfg.N_TASKS

        self.epoch = 0
        self.best_mae = 10 ** 10  # just something high
        self.best_epoch = -1

        self.writer = SummaryWriter(cfg.SAVE_DIR)

        if self.cfg.MAML or self.epoch < self.cfg.ALPHA_START:
            self.meta_wrapper.disable_alpha_updates()

        # if cfg.RESUME:
        #     self.load_state(cfg.RESUME_PATH)
        #     print(f'Resuming from epoch {self.epoch}')
        # else:
        #     self.save_eval_pics()
        #     self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], self.epoch)

        # self.test_functional()

    # def test_functional(self):
    #     """ Just a small function so I can verify with Pycharm debugger if the functional model even works"""
    #
    #     tasks_sampler = iter(self.train_loader)
    #     img, gt, _, _ = next(tasks_sampler)
    #     img = img.squeeze()
    #     model_weights = self.model.state_dict()
    #     # model_pred = self.model.forward(img)
    #     model_func_pred = self.model_funct.forward(img, model_weights, True)
    #     print('done')

    def inner_loop(self, train_data, test_data, theta):
        theta_values = list(theta[k] for k in theta if not k.startswith('alpha.'))
        theta_names = list(k for k in theta if not k.startswith('alpha.'))
        alpha_values = list(theta[k] for k in theta if k.startswith('alpha.'))

        train_loss, train_pred, train_error = self.meta_wrapper.train_forward(train_data, theta)
        grads = torch.autograd.grad(train_loss, theta_values)

        theta_prime = OrderedDict((n, w - a * g) for n, w, a, g in zip(theta_names, theta_values, alpha_values, grads))

        test_loss, test_pred, test_error = self.meta_wrapper.train_forward(test_data, theta_prime)

        before_error = train_error / (self.cfg_data.LABEL_FACTOR * self.cfg_data.K_TRAIN)
        after_error = test_error / (self.cfg_data.LABEL_FACTOR * self.cfg_data.K_META)
        avg_AE = before_error - after_error

        return test_loss, avg_AE

    def run_epoch(self):
        tasks_sampler = iter(self.train_loader)
        items_left = len(self.train_loader.dataset)

        n_improvements = 0
        n_worsenings = 0
        mean_improvements = []
        while items_left > self.n_tasks:
            theta = self.meta_wrapper.get_theta()
            if self.cfg.MAML or self.epoch < self.cfg.ALPHA_START:
                theta_weights = [v for k, v in theta.items() if not k.startswith('alpha.')]
            else:
                theta_weights = list(theta.values())

            total_metaloss = torch.tensor(0).float().cuda()
            total_AEs = 0

            self.meta_optimiser.zero_grad()  # Prob not needed since we set the gradients manually
            for task_idx in range(self.n_tasks):
                train_img, train_gt, test_img, test_gt = next(tasks_sampler)
                items_left -= 1

                train_data = (train_img, train_gt)
                test_data = (test_img, test_gt)
                test_loss, avg_AE = self.inner_loop(train_data, test_data, theta)
                total_metaloss += test_loss
                total_AEs += avg_AE

            avg_metaloss = total_metaloss / self.n_tasks
            metagrads = torch.autograd.grad(avg_metaloss, theta_weights)

            for w, g in zip(theta_weights, metagrads):
                w.grad = g
            self.meta_optimiser.step()

            mean_improvement = (total_AEs / self.n_tasks).item()
            mean_improvements.append(mean_improvement)
            if mean_improvement > 0:
                n_improvements += 1
            else:
                n_worsenings += 1
        return mean_improvements, n_improvements, n_worsenings

    def train(self):  # Outer loop
        self.meta_wrapper.eval()
        # self.evaluate_model()

        self.meta_wrapper.train()
        while self.epoch < self.cfg.MAX_EPOCH:  # While not done
            self.epoch += 1

            if self.epoch == self.cfg.ALPHA_START:
                self.meta_wrapper.enable_alpha_updates()

            # Log alpha stats
            for alpha_name, alpha_value in self.meta_wrapper.base_model.alpha.items():
                alpha_mean = torch.mean(alpha_value).item()
                alpha_std = torch.std(alpha_value).item()

                self.writer.add_scalar(f'Alpha_stats_means/{alpha_name}', alpha_mean, self.epoch)
                self.writer.add_scalar(f'Alpha_stats_stds/{alpha_name}', alpha_std, self.epoch)

            mean_improvements, n_improvements, n_worsenings = self.run_epoch()
            percent_improved = n_improvements / (n_improvements + n_worsenings) * 100
            self.writer.add_scalar('Train/mean_improvement', np.mean(mean_improvements), self.epoch)
            self.writer.add_scalar('Train/n_improvements', n_improvements, self.epoch)
            self.writer.add_scalar('Train/n_worsenings', n_worsenings, self.epoch)
            self.writer.add_scalar('Train/percent_improved', percent_improved, self.epoch)

            print(f'ep {self.epoch} mean test improvement: {np.mean(mean_improvements):.3f}. '
                  f'{percent_improved:.1f}% improved. {n_improvements} improved, {n_worsenings} worsened.')

            if self.epoch % self.cfg.SAVE_EVERY == 0:
                self.save_state()
        self.save_state()

    def evaluate_model(self):
        MLosses = []
        MAEs = []
        MSEs = []
        for val_loader in self.val_loaders:
            adapt_data = val_loader.dataset.get_adapt_batch()

            theta = self.meta_wrapper.get_theta()
            theta_values = list(theta[k] for k in theta if not k.startswith('alpha.'))
            theta_names = list(k for k in theta if not k.startswith('alpha.'))
            alpha_values = list(theta[k] for k in theta if k.startswith('alpha.'))

            train_loss, train_pred, train_error = self.meta_wrapper.train_forward(adapt_data, theta)
            grads = torch.autograd.grad(train_loss, theta_values)

            theta_prime = OrderedDict(
                (n, w - a * g) for n, w, a, g in zip(theta_names, theta_values, alpha_values, grads))

            _MAE = []
            _MSE = []
            _Mloss = []
            for eval_data in val_loader:
                img, pred, gt, loss, AE, SE = self.meta_wrapper.test_forward(eval_data, theta_prime)
                _Mloss.append(loss.item())
                _MAE.append(AE.item() / self.cfg_data.LABEL_FACTOR)
                _MSE.append(SE.item() / self.cfg_data.LABEL_FACTOR)

            MLoss = np.mean(_Mloss)
            MAE = np.mean(_MAE)
            MSE = np.mean(_MSE)

        MLosses.append(MLoss)
        MAEs.append(MAE)
        MSEs.append(MSE)

        return MLosses, MAEs, MSEs

    def save_state(self, name_extra=''):
        if name_extra:
            save_name = f'{self.cfg.STATE_DICTS_DIR}/save_state_ep_{self.epoch}_{name_extra}.pth'
        else:
            save_name = f'{self.cfg.STATE_DICTS_DIR}/save_state_ep_{self.epoch}.pth'

        save_sate = {
            'epoch': self.epoch,
            'best_epoch': self.best_epoch,
            'best_mae': self.best_mae,
            'net': self.meta_wrapper.base_model.state_dict(),
            'optim': self.meta_optimiser.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'save_dir_path': self.cfg.SAVE_DIR,
        }

        torch.save(save_sate, save_name)

    # def load_state(self, state_path):
    #     resume_state = torch.load(state_path)
    #     self.epoch = resume_state['epoch']
    #     self.best_epoch = resume_state['best_epoch']
    #     self.best_mae = resume_state['best_mae']
    #
    #     self.model.load_state_dict(resume_state['net'])
    #     self.meta_optimiser.load_state_dict(resume_state['optim'])
    #     self.scheduler.load_state_dict(resume_state['scheduler'])


def print_fancy_new_best_MAE():
    """ For that extra bit of dopamine rush when you get a new high-score"""

    new_best = '#' + '=' * 20 + '<' * 3 + ' NEW BEST MAE ' + '>' * 3 + '=' * 20 + '#'
    n_chars = len(new_best)
    bar = '#' + '=' * (n_chars - 2) + '#'
    print(bar)
    print(new_best)
    print(bar)
