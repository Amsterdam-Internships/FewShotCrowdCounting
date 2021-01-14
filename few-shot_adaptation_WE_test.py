import numpy as np
import torch
import torch.nn as nn
from models.SCC_Model.CSRNet import CSRNet
from datasets.WE.loading_data import loading_data
from datasets.WE.setting import cfg_data

# MODEL_PATH = 'CSRNet_SGD_STANDARD_100_epochs.pth'
# MODEL_PATH = 'CSRNet_MAML_100_epochs_MAE_10_945.pt'
MODEL_PATH = 'CSRNet_SGD_ep_200_MAE_11.068.pt'
# MODEL_PATH = 'CSRNet_SGD_ep_225_MAE_13.837.pt'
LR = 1e-5


def load_net():
    """A simple function to load a new network from disk"""

    network = CSRNet()
    network.cuda()

    model_path = MODEL_PATH
    my_net = torch.load(model_path)
    for key in list(my_net.keys()):  # Added to be compatible with C3Framework
        new_key = key.replace('CCN.', '')
        my_net[new_key] = my_net.pop(key)
    network.load_state_dict(my_net)  # Note: changed to my_net

    return network


def main():
    network = load_net()
    # for param in network.frontend.parameters():
    #     param.requires_grad = False

    train_dataloader, test_dataloader, restore_transform = loading_data()

    adapt_and_test(network, test_dataloader[3], [60])


def adapt_and_test(network, dataloader, adapt_images):
    optim = torch.optim.SGD(network.parameters(), lr=LR)
    loss_fn = nn.MSELoss().cuda()

    if adapt_images:
        print('adapting network to novel scene')
        for idx, data in enumerate(dataloader):
            if idx in adapt_images:
                img, gt = data
                img = img.cuda()
                gt = gt.cuda().squeeze()

                optim.zero_grad()
                pred = network(img).squeeze()
                loss = loss_fn(pred, gt)
                loss.backward()
                optim.step()
    else:
        print('no adaptation is being performed')

    print('testing network on same novel scene')
    _mae = 0
    _mse = 0
    _loss = 0
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            img, gt = data
            img = img.cuda()
            gt = gt.cuda().squeeze()

            pred = network(img).squeeze()
            loss = loss_fn(pred, gt)

            _loss += loss.item()
            pred_cnt = pred.sum() / cfg_data.LOG_PARA
            pred_cnt = pred_cnt.item()
            gt_cnt = gt.sum() / cfg_data.LOG_PARA
            gt_cnt = gt_cnt.item()
            _mae += abs(pred_cnt - gt_cnt)
            _mse += (pred_cnt - gt_cnt) ** 2

    mae = _mae / len(dataloader)
    mse = _mse / len(dataloader)
    summed_loss = _loss
    mloss = _loss / len(dataloader)
    print(f'mae: {mae:.3f}, mse: {mse:.3f}, summed loss: {summed_loss:.3f}, avg loss: {mloss:.3f}')


if __name__ == '__main__':
    main()
