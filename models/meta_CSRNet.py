import torch
import numpy as np


class MetaCSRNet:
    def __init__(self, base_model, functional_model, criterion):
        self.base_model = base_model
        self.functional_model = functional_model
        self.criterion = criterion

    def train_forward(self, data, weights_dict):
        img, gt = data
        img, gt = img.cuda(), gt.squeeze().cuda()

        pred = self.functional_model.forward(img, weights_dict, training=True)
        pred = pred.squeeze()
        loss = self.criterion(pred, gt)

        avg_abs_error = torch.mean(torch.abs(torch.sum(pred.detach() - gt, dim=(-2, -1))))

        return loss, pred, avg_abs_error

    def test_forward(self):
        pass
