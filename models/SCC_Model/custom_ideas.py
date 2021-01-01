import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(True),
                                   nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
                                   nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True),
                                   # Block 1
                                   nn.Conv2d(256, 256, 3, dilation=1, padding=1), nn.ReLU(True),
                                   nn.Conv2d(256, 256, 3, dilation=2, padding=2), nn.ReLU(True),
                                   nn.Conv2d(256, 256, 3, dilation=3, padding=3), nn.ReLU(True),
                                   nn.Conv2d(256, 256, 3, dilation=4, padding=4), nn.ReLU(True),
                                   nn.Conv2d(256, 256, 3, dilation=5, padding=5), nn.ReLU(True),
                                   # Block 2
                                   nn.Conv2d(256, 128, 3, dilation=1, padding=1), nn.ReLU(True),
                                   nn.Conv2d(128, 128, 3, dilation=2, padding=2), nn.ReLU(True),
                                   nn.Conv2d(128, 128, 3, dilation=3, padding=3), nn.ReLU(True),
                                   # Block 3
                                   nn.Conv2d(128, 128, 3, dilation=1, padding=1), nn.ReLU(True),
                                   nn.Conv2d(128, 128, 3, dilation=2, padding=2), nn.ReLU(True),
                                   nn.Conv2d(128, 128, 3, dilation=3, padding=3), nn.ReLU(True),
                                   # To out
                                   nn.Conv2d(128, 64, 3, dilation=1, padding=1), nn.ReLU(True),
                                   nn.Conv2d(64, 32, 3, dilation=1, padding=1), nn.ReLU(True),
                                   nn.Conv2d(32, 1, 3, dilation=1, padding=1)
                                   )
        self._initialize_weights()

    def forward(self, x):
        out = self.model(x)
        return out

    def _initialize_weights(self):
        self._random_initialize_weights()

    def _random_initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                # nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

