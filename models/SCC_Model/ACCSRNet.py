import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F


class ACCSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(ACCSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.guiding_net_feat = [64, 1]  # Last element should be one
        self.guiding_net_aap_shape = (50, 50)  # Output shape of the adaptive average pool module
        self.frontend = self.make_layers(self.frontend_feat)
        self.backend = self.make_layers(self.backend_feat, in_channels=512, dilation=True, sequential=False)
        self.guiding_net = self.make_guiding_net(self.guiding_net_feat, self.backend_feat)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            self.frontend.load_state_dict(mod.features[0:23].state_dict())

    def forward(self, unl_x, x):
        x = self.frontend(x)
        gbn_params = self.gbn_forward(unl_x)
        x = self.backend_pass(x, gbn_params)
        x = self.output_layer(x)
        x = F.upsample(x, scale_factor=8)
        return x.squeeze()

    def gbn_forward(self, unl_x):
        for layer in self.guiding_net[:-1]:
            unl_x = layer(unl_x)
        unl_x = unl_x.flatten(-2, -1)  # Flatten last two dimensions
        unl_x = self.guiding_net[-1](unl_x)  # Last element is the linear layer
        return unl_x.squeeze()


    def backend_pass(self, x, gbn_params):
        cur_idx = 0

        for i in range(0, len(self.backend_feat) * 2, 2):
            conv_layer = self.backend[i]
            relu_layer = self.backend[i + 1]
            n_channels = self.backend_feat[i // 2]

            means = gbn_params[cur_idx: cur_idx + n_channels]
            std_devs = gbn_params[cur_idx + n_channels: cur_idx + 2 * n_channels]

            x = conv_layer(x)
            x = gbn_bn_operation(x, means, std_devs)
            x = relu_layer(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_guiding_net(self, guiding_net_feat, backend_feat):
        out_size = sum(backend_feat) * 2

        layers = self.make_layers(guiding_net_feat, sequential=False)
        layers.append(nn.AdaptiveAvgPool2d(self.guiding_net_aap_shape))
        flattened_shape = torch.tensor(self.guiding_net_aap_shape).prod()
        layers.append(nn.Linear(flattened_shape.item(), out_size))

        return nn.ModuleList(layers)


    def make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=False, sequential=True):
        if dilation:
            d_rate = 2
        else:
            d_rate = 1
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        if sequential:
            return nn.Sequential(*layers)
        else:
            return nn.ModuleList(layers)


def gbn_bn_operation(x, mean, std):
    "Adjusted from: http://d2l.ai/chapter_convolutional-modern/batch-norm.html"
    eps = 1e-10
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)
    d_mean = x.mean(axis=(0, 2, 3), keepdims=True)
    d_var = ((x - d_mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
    x_hat = (x - d_mean) / torch.sqrt(d_var + eps)
    x_prime = std * (x_hat + mean)

    return x_prime



