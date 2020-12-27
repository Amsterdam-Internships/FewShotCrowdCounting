import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from misc.loss_funcs import cal_lc_loss


class CrowdCounter(nn.Module):
    def __init__(self, net, gpus, loss_funcs, cfg=None):
        super(CrowdCounter, self).__init__()

        self.cfg = cfg
        self.CCN = net
        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()

        supported_loss_funcs = ['MSELoss', 'LCLoss']
        if any([loss_func not in supported_loss_funcs for loss_func in loss_funcs]):
            print(f"One or more of these loss functions are not supported: {loss_funcs}")

        if 'MSELoss' in loss_funcs:
            self.use_mse_loss = True
            self.loss_mse_fn = nn.MSELoss().cuda()

        if 'LCLoss' in loss_funcs:
            self.use_lc_loss = True
        else:
            self.use_lc_loss = False

    @property
    def loss(self):
        return self.summed_loss

    def forward(self, img, gt_map):
        density_map = self.CCN(img)
        self.summed_loss = self.build_loss(density_map.squeeze(), gt_map.squeeze())
        return density_map

    def build_loss(self, density_map, gt_data):
        loss = None  # Silences warning. Should make better if formal
        if self.use_mse_loss:
            self.loss_mse = self.loss_mse_fn(density_map, gt_data)
            loss = self.loss_mse

        if self.use_lc_loss:
            self.loss_lc = self.cfg.LC_FAC * cal_lc_loss(density_map, gt_data)
            loss += self.loss_lc

        return loss

    def test_forward(self, img):
        density_map = self.CCN(img)
        return density_map
