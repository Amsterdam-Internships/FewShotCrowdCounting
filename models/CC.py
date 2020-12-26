import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class CrowdCounter(nn.Module):
    def __init__(self, net, gpus, loss_funcs, cfg=None):
        super(CrowdCounter, self).__init__()

        self.CCN = net
        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()

        supported_loss_funcs = ['MSELoss']
        if any([loss_func not in supported_loss_funcs for loss_func in loss_funcs]):
            print(f"One or more of these loss functions are not supported: {loss_funcs}")

        if 'MSELoss' in loss_funcs:
            self.loss_mse_fn = nn.MSELoss().cuda()


    @property
    def loss(self):
        return self.loss_mse

    def forward(self, img, gt_map):
        density_map = self.CCN(img)
        self.loss_mse = self.build_loss(density_map.squeeze(), gt_map.squeeze())
        return density_map

    def build_loss(self, density_map, gt_data):
        loss_mse = None
        if self.loss_mse_fn:
            loss_mse = self.loss_mse_fn(density_map, gt_data)

        if not loss_mse:
            print("Something went wrong computing the loss.")
            exit(1)

        return loss_mse

    def test_forward(self, img):
        density_map = self.CCN(img)
        return density_map
