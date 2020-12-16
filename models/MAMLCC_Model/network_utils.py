import numpy as np
import torch
from datasets.WE_MAML.setting import cfg_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def forward_pass(network, _in, _tar, mode='validation', weights=None):
    _input = _in.to(device)
    _target = _tar.float().unsqueeze(0).to(device)

    output = network.network_forward(_input, weights)

    if mode == 'validation':
        return [output]
    else:
        loss = network.loss_function(output, _target)
        return [output, loss]


def evaluate(network, dataloader, mode='validation', weights=None):
    mae, mse, loss = 0.0, 0.0, 0.0
    for idx, (_in, _tar) in enumerate(dataloader):
        result = forward_pass(network, _in, _tar, mode, weights)
        pred_cnt = result[0].data.sum() / cfg_data.LOG_PARA
        gt_cnt = _tar.sum().type(torch.FloatTensor).cuda() / cfg_data.LOG_PARA
        difference = pred_cnt - gt_cnt
        _mae = torch.abs(difference)
        _mse = difference ** 2

        mae += _mae.item()
        mse += _mse.item()

        if mode == 'training':
            loss += result[1].item()

    mae /= len(dataloader)
    mse = np.sqrt(mse / len(dataloader))

    if mode == 'training':
        loss /= len(dataloader)
        return (loss, mae, mse)

    return mae, mse
