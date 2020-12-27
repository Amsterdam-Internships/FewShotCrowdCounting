import torch.nn as nn

def cal_lc_loss(pred_den, gt_den, sizes=(1, 2, 4)):
    criterion_L1 = nn.L1Loss()
    Lc_loss = None
    for s in sizes:
        pool = nn.AdaptiveAvgPool2d(s)
        est = pool(pred_den.unsqueeze(0)).squeeze()
        gt = pool(gt_den.unsqueeze(0)).squeeze()

        if Lc_loss:
            Lc_loss += criterion_L1(est, gt)
        else:
            Lc_loss = criterion_L1(est, gt)
    return Lc_loss