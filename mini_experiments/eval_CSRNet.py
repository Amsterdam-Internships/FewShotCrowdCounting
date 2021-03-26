import numpy as np
import torch

from models.CSRNet.CSRNet import CSRNet
from datasets.standard.SHTB_CSRNet.loading_data import loading_data


def evaluate_model(model, dataloader):

    model.eval()
    with torch.no_grad():
        AEs = []  # Absolute Errors
        SEs = []  # Squared Errors


        for idx, (img, gt) in enumerate(dataloader):
            img = img.cuda()
            gt = gt.squeeze().cuda()  # Remove batch dim, insert channel dim

            den = model(img).cpu()
            den = den.squeeze()  # Remove channel dim

            pred_cnt = den.sum()
            gt_cnt = gt.sum()
            AEs.append(torch.abs(pred_cnt - gt_cnt).item())
            SEs.append(torch.square(pred_cnt - gt_cnt).item())
            print(f'pred: {pred_cnt:.3f}, gt: {gt_cnt:.3f}, AE: {AEs[-1]:.3f}, SE: {SEs[-1]:.3f}')

        MAE = np.mean(AEs)
        MSE = np.mean(SEs)

    return MAE, MSE

def main():
    _, test_loader, _ = loading_data(224)

    model = CSRNet()

    resume_state = torch.load('partBmodel_best.pth.tar')
    model.load_state_dict(resume_state['state_dict'])
    model.cuda()

    MAE, MSE = evaluate_model(model, test_loader)
    print(MAE, MSE)


if __name__ == '__main__':
    main()