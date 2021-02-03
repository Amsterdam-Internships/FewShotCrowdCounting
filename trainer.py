import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from datasets.SHTA.settings import cfg_data
from datasets.SHTA.loading_data import loading_data
from datasets.dataset_utils import split_image_and_den, unsplit_den, unsplit_img
import matplotlib.pyplot as plt
from matplotlib import cm as CM


class Trainer:
    def __init__(self, model, cfg):  # Todo: with parameters etc. Like C3
        self.model = model
        self.cfg = cfg
        self.cfg_data = cfg_data

        self.train_loader, self.test_loader, self.restore_transform = loading_data()
        self.train_samples = len(self.train_loader.dataset)
        self.test_samples = len(self.test_loader.dataset)
        self.eval_save_example_every = self.test_samples // self.cfg.SAVE_NUM_EVAL_EXAMPLES

        self.criterion = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=cfg.LR_GAMMA)

        self.epoch = 0
        self.best_mae = 10 ** 10  # just something high
        self.best_epoch = -1

        self.writer = SummaryWriter(cfg.SAVE_DIR)

        if cfg.RESUME:
            self.load_state(cfg.RESUME_PATH)
            print(f'Resuming from epoch {self.epoch}')
        # else:
        #     self.save_eval_pics()

    def train(self):
        # MAE = self.evaluate_model()
        # print(f'Initial MAE: {MAE:.3f}')
        self.model.train()

        while self.epoch < self.cfg.MAX_EPOCH:
            self.epoch += 1

            epoch_start_time = time.time()
            losses, errors, last_out_den, last_gts = self.run_epoch()
            epoch_run_time = time.time() - epoch_start_time

            MAE = torch.mean(torch.stack(errors)) / (self.cfg_data.TRAIN_BS * self.cfg_data.LABEL_FACTOR)
            avg_loss = np.mean(losses)
            pred_cnt = last_out_den[0].detach().cpu().sum() / self.cfg_data.LABEL_FACTOR
            gt_cnt = last_gts[0].cpu().sum() / self.cfg_data.LABEL_FACTOR
            print(f'ep {self.epoch}: Average loss={avg_loss:.3f}, Patch MAE={MAE:.3f}.'
                  f'  Example: pred={pred_cnt:.3f}, gt={gt_cnt:.3f}. Train time: {epoch_run_time:.3f}')

            if self.epoch % self.cfg.EVAL_EVERY == 0:
                eval_MAE = self.evaluate_model()
                if eval_MAE < self.best_mae:
                    self.best_mae = eval_MAE
                    print_fancy_new_best_MAE()
                elif self.epoch % self.cfg.SAVE_EVERY == 0:
                    self.save_state(f'MAE_{MAE:.3f}')
                print(f'MAE: {eval_MAE:.3f}, best MAE: {self.best_mae:.3f}')

            if self.epoch in self.cfg.LR_STEP_EPOCHS:
                self.scheduler.step()
                print(f'Learning rate adjusted to {self.scheduler.get_last_lr()} at epoch {self.epoch}')

    def run_epoch(self):
        losses = []
        errors = []

        out_den = None  # SILENCE WENCH!
        gts = None  # silences the 'might not be defined' warning below the for loop.

        for idx, (images, gts) in enumerate(self.train_loader):
            images = images.cuda()
            gts = gts.cuda()

            self.optim.zero_grad()
            out_den, out_count = self.model(images)
            out_den = out_den.squeeze()
            loss = self.criterion(out_den, gts)
            loss.backward()
            self.optim.step()

            losses.append(loss.cpu().item())
            errors.append(torch.abs(torch.sum(out_den - gts, axis=(1, 2))).sum())

        # Also return the last predicted densities and corresponding gts. This allows for informative prints
        return losses, errors, out_den, gts

    def evaluate_model(self):

        plt.cla()  # Clear plot for new ones
        self.model.eval()
        with torch.no_grad():
            errors = []

            abs_patch_errors = torch.zeros(cfg_data.PATCH_SIZE, cfg_data.PATCH_SIZE)
            summed_patch_errors = torch.zeros(self.model.n_patches, self.model.n_patches)

            for idx, (img_patches, gt_patches, img_resolution) in enumerate(self.test_loader):
                img_patches = img_patches.squeeze().cuda()
                gt_patches = gt_patches.squeeze()
                img_resolution = img_resolution.squeeze()

                pred_den, pred_count = self.model(img_patches)
                pred_den = pred_den.squeeze().cpu()

                den, gt = unsplit_den(pred_den, gt_patches, img_resolution)

                pred_cnt = den.sum() / self.cfg_data.LABEL_FACTOR
                gt_cnt = gt.sum() / self.cfg_data.LABEL_FACTOR
                errors.append(torch.abs(pred_cnt - gt_cnt))

                if idx % self.eval_save_example_every == 0:
                    plt.imshow(den, cmap=CM.jet)
                    save_path = os.path.join(self.cfg.PICS_DIR, f'pred_{idx}_ep_{self.epoch}.jpg')
                    plt.title(f'Predicted count: {pred_cnt:.3f} (GT: {gt_cnt})')
                    plt.savefig(save_path)

                abs_patch_errors += torch.sum(torch.abs(gt_patches - pred_den), axis=0)
            for i in range(14):
                for j in range(14):
                    lf = self.cfg_data.LABEL_FACTOR  # So next line fits on 1 line
                    summed_patch_errors[i, j] = abs_patch_errors[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16].sum() / lf

            MAE = torch.mean(torch.stack(errors))

        plt.cla()
        plt.imshow(abs_patch_errors)
        save_path = os.path.join(self.cfg.PICS_DIR, f'errors_ep_{self.epoch}.jpg')
        plt.savefig(save_path)
        plt.imshow(summed_patch_errors)
        save_path = os.path.join(self.cfg.PICS_DIR, f'summed_patch_ep_{self.epoch}.jpg')
        plt.savefig(save_path)


        return MAE

    def save_eval_pics(self):
        for idx, (img_patches, gt_patches, img_resolution) in enumerate(self.test_loader):
            img_patches = img_patches.squeeze().cuda()
            gt_patches = gt_patches.squeeze()
            img_resolution = img_resolution.squeeze()

            # why create an extra function when you can do this? TODO: make more general function
            _, gt = unsplit_den(gt_patches, gt_patches, img_resolution)

            if idx % self.eval_save_example_every == 0:
                img = unsplit_img(img_patches, img_resolution)
                img = self.restore_transform(img)
                gt_count = gt.sum() / self.cfg_data.LABEL_FACTOR
                gt_count = torch.round(gt_count)

                plt.imshow(img)
                save_path = os.path.join(self.cfg.PICS_DIR, f'img_{idx}.jpg')
                plt.title(f'GT count: {gt_count}')
                plt.savefig(save_path)

                plt.imshow(gt, cmap=CM.jet)
                save_path = os.path.join(self.cfg.PICS_DIR, f'gt_{idx}.jpg')
                plt.title(f'GT count: {gt_count}')
                plt.savefig(save_path)
        plt.cla()

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
            'optim': self.optim.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'exp_path': self.cfg.SAVE_DIR,
        }

        torch.save(save_sate, save_name)

    def load_state(self, state_path):
        resume_state = torch.load(state_path)
        self.epoch = resume_state['epoch']
        self.best_epoch = resume_state['best_epoch']
        self.best_mae = resume_state['best_mae']

        self.model.load_state_dict(resume_state['net'])
        self.optim.load_state_dict(resume_state['optim'])
        self.scheduler.load_state_dict(resume_state['scheduler'])


def print_fancy_new_best_MAE():
    """ For that extra bit of dopamine rush when you get a new high-score"""

    new_best = '#' + '=' * 20 + '<' * 3 + ' NEW BEST MAE ' + '>' * 3 + '=' * 20 + '#'
    n_chars = len(new_best)
    bar = '#' + '=' * (n_chars - 2) + '#'
    print(bar)
    print(new_best)
    print(bar)
